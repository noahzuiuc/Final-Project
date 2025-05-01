#!/h/haoran/anaconda3/bin/python
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import argparse
import Constants
import torch
import torch.nn as nn
from torch.utils import data
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
from run_classifier_dataset_utils import InputExample, convert_examples_to_features
from pathlib import Path
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from gradient_reversal import GradientReversal
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss, mean_squared_error, classification_report
import random
import json
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from utils import create_hdf_key, Classifier, get_emb_size, MIMICDataset, extract_embeddings, EarlyStopping, load_checkpoint
from sklearn.model_selection import ParameterGrid

# --- Apex Import ---
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    print("Warning: NVIDIA Apex not found. Mixed precision training will be disabled. "
          "Install Apex from https://github.com/NVIDIA/apex for potential memory savings and speedup.")
    APEX_AVAILABLE = False
# --- End Apex Import ---


parser = argparse.ArgumentParser('Fine-tunes a pre-trained BERT model on a certain target for one fold. Outputs fine-tuned BERT model and classifier, ' +
                                 'as well as a pickled dictionary mapping id: predicted probability')
# --- Keep all argparse arguments the same ---
parser.add_argument("--df_path",help = 'must have the following columns: seqs, num_seqs, fold, with note_id as index', type=str)
parser.add_argument("--model_path", type=str)
# ... (rest of parser arguments remain unchanged) ...
parser.add_argument('--overwrite', help = 'whether to overwrite existing model/predictions', action = 'store_true')
args = parser.parse_args()

if os.path.isfile(os.path.join(args.output_dir, 'preds.pkl')) and not args.overwrite:
    print("File already exists; exiting.")
    sys.exit()

print('Reading dataframe...', flush = True)
df = pd.read_pickle(args.df_path)
if 'note_id' in df.columns:
    df = df.set_index('note_id')

tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertModel.from_pretrained(args.model_path)

target = args.target_col_name
assert(target in df.columns)

#even if no adversary, must have valid protected group column for code to work
if args.use_adversary:
    protected_group = args.protected_group
    assert(protected_group in df.columns)
    if args.use_new_mapping:
        mapping = Constants.newmapping
        for i in Constants.drop_groups[protected_group]:
            df = df[df[protected_group] != i]
    else:
        mapping = Constants.mapping

other_fields_to_include = args.other_fields
if args.freeze_bert:
    print("Freezing BERT model parameters.", flush=True)
    for param in model.parameters():
        param.requires_grad = False

assert('fold' in df.columns)
for i in args.fold_id:
    assert(i in df['fold'].unique())
assert('test' in df['fold'].unique())
fold_id = args.fold_id

if args.gridsearch_c:
    assert(args.task_type == 'binary')
    c_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000]
else:
    c_grid = [2]

Path(args.output_dir).mkdir(parents = True, exist_ok = True)

EMB_SIZE = get_emb_size(args.emb_method)
train_df = df[~df.fold.isin(['test', 'NA', *fold_id])]
val_df = df[df.fold.isin(fold_id)]
test_df = df[df.fold == 'test']

# --- convert_input_example, EmbFeature, Embdataset, Discriminator remain unchanged ---
# ... (Keep the class definitions as before, ensuring Embdataset handles y shape correctly) ...
class Embdataset(data.Dataset):
    def __init__(self, features, gen_type):
        self.features = features #list of EmbFeatures
        self.gen_type = gen_type
        self.length = len(features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        emb = torch.tensor(self.features[index].emb, dtype = torch.float32)
        if args.task_type in ['binary', 'regression']:
            # Ensure target is float
            y_val = float(self.features[index].y) if self.features[index].y is not None else 0.0
            y = torch.tensor(y_val, dtype = torch.float32)
        else: # multiclass
            # Ensure target is long
            y_val = int(self.features[index].y) if self.features[index].y is not None else 0
            y = torch.tensor(y_val, dtype = torch.long)

        # Ensure binary targets are shaped correctly for BCEWithLogitsLoss: [batch_size, 1]
        if args.task_type == 'binary':
            y = y.view(-1, 1) # Reshape to [N, 1] - This should happen in DataLoader collation if batching

        other_fields = self.features[index].other_fields
        guid = self.features[index].guid

        # Need to handle list of tensors for other_fields if batching
        # Let's assume other_fields are processed correctly later or are simple types for now
        return emb, y, guid, other_fields # Return other_fields as list

class Discriminator(nn.Module): # --- Unchanged ---
    def __init__(self, input_dim, num_layers, num_categories, lm):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        assert(num_layers >= 1)
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.lm = lm
        self.layers = [GradientReversal(lambda_ = lm)]
        current_dim = input_dim
        for c, i in enumerate(range(num_layers)):
            if c != num_layers-1:
                next_dim = current_dim // 2
                self.layers.append(nn.Linear(current_dim, next_dim))
                self.layers.append(nn.ReLU())
                current_dim = next_dim
            else:
                # Final layer outputs logits for num_categories
                self.layers.append(nn.Linear(current_dim, num_categories))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x # Output raw logits

# --- Grid definition remains unchanged ---
# ... (grid definition code) ...
if args.gridsearch_classifier:
    assert(args.freeze_bert)
    grid = list(ParameterGrid({
        'num_layers': [2,3,4],
        'dropout_prob': [0, 0.2],
        'decay_rate': [2,4,6] # Assuming decay_rate is used within Classifier
    }))
    grid.append({
        'num_layers': 1,
        'dropout_prob': 0,
        'decay_rate': 2
        })
    for i in grid: # adds extra fields to input arguments
        i['input_dim'] = EMB_SIZE + len(other_fields_to_include)
        i['task_type'] = args.task_type
else:
    grid = [{ # only one parameter combination
        'input_dim': EMB_SIZE + len(other_fields_to_include),
        'num_layers': args.predictor_layers,
        'dropout_prob': args.dropout,
        'task_type': args.task_type
        # Assuming decay_rate is not needed if not gridsearching, or add default
    }]

if args.task_type == 'multiclass':
    n_classes = len(df[target].unique())
    print(f"Multiclass task detected with {n_classes} classes.", flush=True)
    for i in grid:
        i['multiclass_nclasses'] = n_classes


if args.use_adversary:
    # Calculate num_categories based on the mapping used
    num_adv_categories = len(mapping[protected_group]) if protected_group in mapping else len(df[protected_group].unique())
    print(f"Adversary using {num_adv_categories} categories for group '{protected_group}'.", flush=True)
    # Adversary input dim depends on fairness def
    adv_input_dim = EMB_SIZE
    if args.fairness_def == 'odds':
         adv_input_dim += 1 # Add space for the target variable y
    discriminator = Discriminator(adv_input_dim, args.adv_layers, num_adv_categories, args.lm)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs: {n_gpu}", flush=True)

model.to(device)
if args.use_adversary:
    discriminator.to(device)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

# --- Use BCEWithLogitsLoss --- (Keep this change)
if args.task_type == 'binary':
    criterion = nn.BCEWithLogitsLoss()
elif args.task_type == 'multiclass':
    criterion = nn.CrossEntropyLoss()
elif args.task_type == 'regression':
    criterion = nn.MSELoss()
# --- End Use BCEWithLogitsLoss ---

criterion_adv = nn.CrossEntropyLoss() # Expects raw logits from Discriminator


# --- DataParallel wrapping BEFORE Apex initialization ---
if n_gpu > 1:
    print("Using DataParallel...", flush=True)
    model = torch.nn.DataParallel(model)
    # Predictor will be wrapped later inside the loop
    if args.use_adversary:
        discriminator = torch.nn.DataParallel(discriminator)

# --- get_embs function remains largely unchanged, just remove autocast ---
def get_embs(generator):
    features = []
    model.eval()
    with torch.no_grad():
        # No autocast here
        for input_ids, input_mask, segment_ids, y, group, guid, other_vars in tqdm(generator, desc="Generating Embeddings"):
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            bert_out = extract_embeddings(hidden_states, args.emb_method)

            y = y.cpu()
            group = group.cpu()

            for c,i in enumerate(guid):
                 note_id, seq_id = i.split('-')
                 emb = bert_out[c,:].detach().cpu().numpy()
                 current_y = y[c].item() if y[c].numel() == 1 else y[c].numpy()
                 current_group = group[c].item() if group[c].numel() == 1 else group[c].numpy()
                 current_other_fields = []
                 for ov_list in other_vars: # other_vars is now potentially a list of lists/tensors
                    if isinstance(ov_list, torch.Tensor):
                        val = ov_list[c].item() if ov_list[c].numel() == 1 else ov_list[c].numpy()
                    else: # Assume simple list
                        val = ov_list[c]
                    current_other_fields.append(val)

                 features.append(EmbFeature(emb = emb, y = current_y, guid = i, group = current_group, other_fields= current_other_fields))

    return features


print('Featurizing examples...', flush = True)
# --- Data loading remains unchanged ---
# ... (data loading code) ...
if not args.pregen_emb_path:
    features_train = convert_examples_to_features(examples_train,
                                                Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))
    features_eval = convert_examples_to_features(examples_eval,
                                               Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))
    features_test = convert_examples_to_features(examples_test,
                                               Constants.MAX_SEQ_LEN, tokenizer, output_mode = ('regression' if args.task_type == 'regression' else 'classification'))

    training_set = MIMICDataset(features_train, 'train' ,args.task_type)
    training_generator = data.DataLoader(training_set, shuffle = True,  batch_size = args.train_batch_size, drop_last = True)
    val_set = MIMICDataset(features_eval, 'val', args.task_type)
    val_generator = data.DataLoader(val_set, shuffle = False,  batch_size = args.train_batch_size)
    test_set = MIMICDataset(features_test, 'test', args.task_type)
    test_generator = data.DataLoader(test_set, shuffle = False,  batch_size = args.train_batch_size)

if args.freeze_bert: #only need to precalculate for training and val set
    if args.pregen_emb_path:
        print(f"Loading pregenerated embeddings from: {args.pregen_emb_path}", flush=True)
        pregen_embs = pickle.load(open(args.pregen_emb_path, 'rb'))
        features_train_embs = convert_examples_to_features_emb(examples_train, pregen_embs)
        features_val_embs = convert_examples_to_features_emb(examples_eval, pregen_embs)
        features_test_embs = convert_examples_to_features_emb(examples_test, pregen_embs)
    else:
        print("Calculating embeddings for frozen BERT...", flush=True)
        # Need a temporary generator without shuffling for consistency if using pregen embs
        temp_train_gen = data.DataLoader(training_set, shuffle = False, batch_size=args.train_batch_size)
        temp_val_gen = data.DataLoader(val_set, shuffle = False, batch_size=args.train_batch_size)
        temp_test_gen = data.DataLoader(test_set, shuffle = False, batch_size=args.train_batch_size)
        features_train_embs = get_embs(temp_train_gen)
        features_val_embs = get_embs(temp_val_gen)
        features_test_embs = get_embs(temp_test_gen)
        print("Embeddings calculated.", flush=True)

    # Use Embdataset with precalculated embeddings
    training_generator = data.DataLoader(Embdataset(features_train_embs, 'train'), shuffle = True, batch_size = args.train_batch_size, drop_last = True)
    val_generator = data.DataLoader(Embdataset(features_val_embs, 'val'), shuffle = False,  batch_size = args.train_batch_size)
    test_generator= data.DataLoader(Embdataset(features_test_embs, 'test'), shuffle = False,  batch_size = args.train_batch_size)


num_train_epochs = args.max_num_epochs
learning_rate = args.lr
print(f"Training Generator Length: {len(training_generator)}", flush=True)
if len(training_generator) == 0:
    raise ValueError("Training generator has length 0. Check data splitting and batch size.")
num_train_optimization_steps = len(training_generator) * num_train_epochs
if num_train_optimization_steps <= 0 :
     raise ValueError("num_train_optimization_steps must be > 0. Check len(training_generator) and num_train_epochs.")

warmup_proportion = 0.1

PREDICTOR_CHECKPOINT_PATH = os.path.join(args.output_dir, 'predictor.chkpt')
MODEL_CHECKPOINT_PATH = os.path.join(args.output_dir, 'model.chkpt')

grid_auprcs = []
es_models = []
optimal_cs = []
actual_val = val_df[target]

# --- merge_probs, avg_probs, etc. remain unchanged ---
# ... (Keep helper functions as before) ...
def merge_probs(probs, c):
    if not probs: return 0.5
    probs_arr = np.array(probs)
    # Add small epsilon to handle potential division by zero if c is small and len is large
    c = max(c, 1e-6)
    denominator = (1 + len(probs_arr) / float(c))
    if denominator == 0: return np.mean(probs_arr) # Fallback if denominator is zero
    return (np.max(probs_arr) + np.mean(probs_arr) * len(probs_arr) / float(c)) / denominator

def avg_probs(probs):
    if not probs: return 0.5
    return np.mean(probs)

def avg_probs_multiclass(probs_list):
    if not probs_list: return 0
    valid_probs = [p for p in probs_list if p is not None and isinstance(p, np.ndarray)]
    if not valid_probs: return 0
    mean_probs = np.mean(np.array(valid_probs), axis=0)
    return np.argmax(mean_probs)

def merge_regression(preds):
    if not preds: return 0.0
    return np.mean(preds)


# --- evaluate_on_set: Apply sigmoid/softmax AFTER predictor, remove autocast ---
def evaluate_on_set(generator, predictor, emb_gen = False, c_val=2):
    model.eval()
    predictor.eval()
    if generator.dataset.gen_type == 'val':
        df_to_use = val_df
    elif generator.dataset.gen_type == 'test':
        df_to_use = test_df
    elif generator.dataset.gen_type == 'train':
         df_to_use = train_df
    else:
        raise ValueError("Unknown generator type")

    prediction_dict = {str(idx): [None]*row['num_seqs'] for idx, row in df_to_use.iterrows()}
    embs = {str(idx):np.zeros(shape = (row['num_seqs'], EMB_SIZE)) for idx, row in df_to_use.iterrows()}

    desc = f"Evaluating {generator.dataset.gen_type}"
    with torch.no_grad():
        # No autocast wrapper here
        if emb_gen: # Using pre-calculated embeddings
            for batch_embs, y, guid, other_vars in tqdm(generator, desc=f"{desc} (Emb)"):
                batch_embs = batch_embs.to(device)

                predictor_input = batch_embs
                # --- Handle other_vars concatenation ---
                current_batch_other_fields = []
                if other_vars:
                    # Assume other_vars is a list of lists/values for the batch
                    # Transpose if necessary: from list-of-features to feature-of-lists
                    num_other = len(other_vars[0]) # Number of other features per example
                    batch_size = len(other_vars)
                    temp_other_vars = [[] for _ in range(num_other)]
                    for example_vars in other_vars:
                        for feat_idx, val in enumerate(example_vars):
                            temp_other_vars[feat_idx].append(val)

                    for feat_list in temp_other_vars:
                        ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                        if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                        predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                        current_batch_other_fields.append(feat_list) # Keep track for saving embs if needed
                # --- End handle other_vars ---

                output = predictor(predictor_input) # Get raw logits/values

                # --- Convert output to probabilities/values ---
                if args.task_type == 'binary':
                    preds = torch.sigmoid(output).detach().cpu()
                elif args.task_type == 'multiclass':
                    preds = torch.softmax(output, dim=1).detach().cpu()
                else: # Regression
                    preds = output.detach().cpu()

                # --- Store predictions and embeddings ---
                for c, i in enumerate(guid):
                     note_id, seq_id = i.split('-')
                     if note_id not in prediction_dict: continue
                     seq_id_int = int(seq_id)
                     max_seqs = len(prediction_dict[note_id])
                     if seq_id_int < max_seqs:
                         if args.task_type in ['binary', 'regression']:
                             prediction_dict[note_id][seq_id_int] = preds[c].item()
                         else: # multiclass
                             prediction_dict[note_id][seq_id_int] = preds[c,:].numpy()

                         # Store original embedding
                         if seq_id_int < embs[note_id].shape[0]:
                              embs[note_id][seq_id_int, :] = batch_embs[c,:].detach().cpu().numpy()


        else: # Processing text input directly
            for input_ids, input_mask, segment_ids, y, group, guid, other_vars in tqdm(generator, desc=f"{desc} (Text)"):
                input_ids = input_ids.to(device)
                segment_ids = segment_ids.to(device)
                input_mask = input_mask.to(device)

                hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
                bert_out = extract_embeddings(hidden_states, args.emb_method)

                predictor_input = bert_out
                # --- Handle other_vars concatenation (repeated logic, consider helper function) ---
                current_batch_other_fields = []
                if other_vars:
                    num_other = len(other_vars[0]) # Number of other features per example
                    batch_size = len(other_vars)
                    temp_other_vars = [[] for _ in range(num_other)]
                    for example_vars in other_vars:
                        for feat_idx, val in enumerate(example_vars):
                            temp_other_vars[feat_idx].append(val)

                    for feat_list in temp_other_vars:
                        ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                        if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                        predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                        current_batch_other_fields.append(feat_list)
                # --- End handle other_vars ---

                output = predictor(predictor_input) # Raw logits/values

                # --- Convert output to probabilities/values ---
                if args.task_type == 'binary':
                    preds = torch.sigmoid(output).detach().cpu()
                elif args.task_type == 'multiclass':
                    preds = torch.softmax(output, dim=1).detach().cpu()
                else: # Regression
                    preds = output.detach().cpu()

                # --- Store predictions and embeddings ---
                for c, i in enumerate(guid):
                     note_id, seq_id = i.split('-')
                     if note_id not in prediction_dict: continue
                     seq_id_int = int(seq_id)
                     max_seqs = len(prediction_dict[note_id])
                     if seq_id_int < max_seqs:
                         if args.task_type in ['binary', 'regression']:
                             prediction_dict[note_id][seq_id_int] = preds[c].item()
                         else: # multiclass
                             prediction_dict[note_id][seq_id_int] = preds[c,:].numpy()

                         # Store BERT embedding
                         if seq_id_int < embs[note_id].shape[0]:
                              embs[note_id][seq_id_int, :] = bert_out[c,:].detach().cpu().numpy()


    merged_preds = merge_preds(prediction_dict, c_val)
    return (prediction_dict, merged_preds, embs)


# --- Main Training Loop ---
for predictor_params in grid:
    print("\nTesting Predictor Params:", flush = True)
    print(predictor_params, flush = True)
    # Initialize predictor for this grid search iteration
    predictor = Classifier(**predictor_params).to(device)

    # --- DataParallel wrapping for predictor BEFORE Apex initialization ---
    if n_gpu > 1:
        print("Wrapping predictor in DataParallel...", flush=True)
        predictor = torch.nn.DataParallel(predictor)

    # Define parameters to optimize based on freezing/adversary
    param_optimizer_list = []
    models_for_amp = [] # Keep track of models passed to amp

    if not args.freeze_bert:
        param_optimizer_list.extend(list(model.named_parameters()))
        models_for_amp.append(model)
    # Predictor is always trained
    param_optimizer_list.extend(list(predictor.named_parameters()))
    models_for_amp.append(predictor)

    if args.use_adversary and not args.freeze_bert:
        param_optimizer_list.extend(list(discriminator.named_parameters()))
        models_for_amp.append(discriminator)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer_list if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer_list if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
            ]

    # Filter out empty groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g['params']]
    if not optimizer_grouped_parameters:
         raise ValueError("No parameters with requires_grad=True found for the optimizer.")

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

    # --- Apex AMP Initialization ---
    if APEX_AVAILABLE and device.type == 'cuda':
        print(f"Initializing Apex AMP with opt_level=O1 for {len(models_for_amp)} model(s)...", flush=True)
        # Ensure models_for_amp contains only models with params in the optimizer
        models_for_amp, optimizer = amp.initialize(
            models_for_amp, # Pass the list of models
            optimizer,
            opt_level="O1" # O1 is standard mixed precision
        )
        print("Apex AMP initialized.", flush=True)
    elif device.type == 'cuda':
        print("Apex not available, proceeding without mixed precision.", flush=True)
    # --- End Apex AMP Initialization ---

    es = EarlyStopping(patience = args.es_patience)

    for epoch in range(1, num_train_epochs+1):
        # --- Training Phase ---
        if not args.freeze_bert: model.train()
        else: model.eval()
        predictor.train()
        if args.use_adversary and not args.freeze_bert: discriminator.train()

        running_loss = 0.0
        num_steps = 0
        optimizer.zero_grad()

        desc = f"Epoch {epoch} (Train)"
        with tqdm(total=len(training_generator), desc=desc) as pbar:
            if not args.freeze_bert: # Fine-tuning BERT
                for step, batch in enumerate(training_generator):
                    input_ids, input_mask, segment_ids, y, group, _, other_vars = batch
                    input_ids = input_ids.to(device)
                    segment_ids = segment_ids.to(device)
                    input_mask = input_mask.to(device)
                    y = y.to(device)
                    group = group.to(device)

                    # No autocast needed here with Apex O1

                    hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
                    bert_out = extract_embeddings(hidden_states, args.emb_method)

                    predictor_input = bert_out
                    # --- Handle other_vars ---
                    temp_other_vars_tensors = []
                    if other_vars:
                         num_other = len(other_vars[0])
                         batch_size = len(other_vars)
                         temp_other_vars = [[] for _ in range(num_other)]
                         for example_vars in other_vars:
                              for feat_idx, val in enumerate(example_vars):
                                   temp_other_vars[feat_idx].append(val)
                         for feat_list in temp_other_vars:
                              ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                              if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                              predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                              temp_other_vars_tensors.append(ov_tensor)
                    # --- End handle other_vars ---

                    output = predictor(predictor_input) # Raw logits/values

                    # --- Calculate Loss (Ensure y shape is correct) ---
                    if args.task_type == 'binary' and y.ndim == 1:
                        y_target = y.float().view(-1, 1)
                    elif args.task_type == 'multiclass' and y.ndim > 1 and y.shape[-1] == 1:
                        y_target = y.long().squeeze(-1)
                    elif args.task_type == 'multiclass':
                         y_target = y.long() # Expects [N]
                    else: # regression
                        y_target = y.float()
                        if y_target.ndim == 1: y_target = y_target.view(-1,1) # MSE usually wants [N,1]
                        if output.ndim == 1 : output = output.view(-1,1) # Ensure output matches

                    loss = criterion(output, y_target)
                    # --- End Calculate Loss ---


                    adv_loss = torch.tensor(0.0).to(device)
                    if args.use_adversary and not args.freeze_bert:
                        adv_input = bert_out
                        if args.fairness_def == 'odds':
                             y_adv_input = y.float().view(-1, 1) if y.ndim == 1 else y.float()
                             adv_input = torch.cat([adv_input, y_adv_input], 1)
                        adv_pred_logits = discriminator(adv_input)
                        adv_loss = criterion_adv(adv_pred_logits, group.long())

                    total_loss = loss + adv_loss

                    if n_gpu > 1: total_loss = total_loss.mean()

                    # --- Apex Loss Scaling ---
                    if APEX_AVAILABLE and device.type == 'cuda':
                        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        total_loss.backward()
                    # --- End Apex Loss Scaling ---

                    optimizer.step()
                    optimizer.zero_grad()

                    num_steps += 1
                    running_loss += total_loss.item()
                    mean_loss = running_loss / num_steps
                    pbar.update(1)
                    pbar.set_postfix_str("Running Training Loss: %.5f" % mean_loss)

            else: # Training only the classifier (BERT frozen)
                 for step, batch in enumerate(training_generator):
                    batch_embs, y, _, other_vars = batch
                    batch_embs = batch_embs.to(device)
                    y = y.to(device)

                    # No autocast

                    predictor_input = batch_embs
                    # --- Handle other_vars ---
                    temp_other_vars_tensors = []
                    if other_vars:
                        num_other = len(other_vars[0])
                        batch_size = len(other_vars)
                        temp_other_vars = [[] for _ in range(num_other)]
                        for example_vars in other_vars:
                             for feat_idx, val in enumerate(example_vars):
                                  temp_other_vars[feat_idx].append(val)
                        for feat_list in temp_other_vars:
                             ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                             if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                             predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                             temp_other_vars_tensors.append(ov_tensor)
                    # --- End handle other_vars ---

                    output = predictor(predictor_input)

                    # --- Calculate Loss (Ensure y shape is correct) ---
                    if args.task_type == 'binary' and y.ndim == 1:
                        y_target = y.float().view(-1, 1)
                    elif args.task_type == 'multiclass' and y.ndim > 1 and y.shape[-1] == 1:
                         y_target = y.long().squeeze(-1)
                    elif args.task_type == 'multiclass':
                         y_target = y.long()
                    else: # regression
                        y_target = y.float()
                        if y_target.ndim == 1: y_target = y_target.view(-1,1)
                        if output.ndim == 1 : output = output.view(-1,1)
                    loss = criterion(output, y_target)
                    # --- End Calculate Loss ---

                    if n_gpu > 1: loss = loss.mean()

                    # --- Apex Loss Scaling ---
                    if APEX_AVAILABLE and device.type == 'cuda':
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # --- End Apex Loss Scaling ---

                    optimizer.step()
                    optimizer.zero_grad()

                    num_steps += 1
                    running_loss += loss.item()
                    mean_loss = running_loss / num_steps
                    pbar.update(1)
                    pbar.set_postfix_str("Running Training Loss: %.5f" % mean_loss)

        # --- Validation Phase ---
        model.eval()
        predictor.eval()
        if args.use_adversary and not args.freeze_bert: discriminator.eval()

        val_loss = 0
        with torch.no_grad():
            # Checkpoints to save
            checkpoints = {PREDICTOR_CHECKPOINT_PATH: predictor}
            if not args.freeze_bert:
                checkpoints[MODEL_CHECKPOINT_PATH] = model

            # No autocast wrapper for validation
            desc = f"Epoch {epoch} (Val)"
            if args.freeze_bert: # Use pre-calculated embeddings
                for batch_embs, y, guid, other_vars in tqdm(val_generator, desc=f"{desc} Emb)"):
                    batch_embs = batch_embs.to(device)
                    y = y.to(device)

                    predictor_input = batch_embs
                    # --- Handle other_vars ---
                    temp_other_vars_tensors = []
                    if other_vars:
                        num_other = len(other_vars[0])
                        batch_size = len(other_vars)
                        temp_other_vars = [[] for _ in range(num_other)]
                        for example_vars in other_vars:
                             for feat_idx, val in enumerate(example_vars):
                                  temp_other_vars[feat_idx].append(val)
                        for feat_list in temp_other_vars:
                             ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                             if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                             predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                             temp_other_vars_tensors.append(ov_tensor)
                    # --- End handle other_vars ---

                    output = predictor(predictor_input)

                    # --- Calculate Loss ---
                    if args.task_type == 'binary' and y.ndim == 1: y_target = y.float().view(-1, 1)
                    elif args.task_type == 'multiclass' and y.ndim > 1 and y.shape[-1] == 1: y_target = y.long().squeeze(-1)
                    elif args.task_type == 'multiclass': y_target = y.long()
                    else:
                        y_target = y.float()
                        if y_target.ndim == 1: y_target = y_target.view(-1,1)
                        if output.ndim == 1 : output = output.view(-1,1)
                    loss = criterion(output, y_target)
                    # --- End Calculate Loss ---

                    if n_gpu > 1: loss = loss.mean()
                    val_loss += loss.item()

            else: # Process text input directly
                for input_ids, input_mask, segment_ids, y, group, guid, other_vars in tqdm(val_generator, desc=f"{desc} Text)"):
                    input_ids = input_ids.to(device)
                    segment_ids = segment_ids.to(device)
                    input_mask = input_mask.to(device)
                    y = y.to(device)
                    # group = group.to(device) # Only needed if calculating adv loss in val

                    hidden_states, _ = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
                    bert_out = extract_embeddings(hidden_states, args.emb_method)

                    predictor_input = bert_out
                    # --- Handle other_vars ---
                    temp_other_vars_tensors = []
                    if other_vars:
                        num_other = len(other_vars[0])
                        batch_size = len(other_vars)
                        temp_other_vars = [[] for _ in range(num_other)]
                        for example_vars in other_vars:
                             for feat_idx, val in enumerate(example_vars):
                                  temp_other_vars[feat_idx].append(val)
                        for feat_list in temp_other_vars:
                             ov_tensor = torch.tensor(feat_list, dtype=torch.float32).to(device)
                             if ov_tensor.ndim == 1: ov_tensor = ov_tensor.unsqueeze(1)
                             predictor_input = torch.cat([predictor_input, ov_tensor], 1)
                             temp_other_vars_tensors.append(ov_tensor)
                    # --- End handle other_vars ---

                    output = predictor(predictor_input)

                    # --- Calculate Loss ---
                    if args.task_type == 'binary' and y.ndim == 1: y_target = y.float().view(-1, 1)
                    elif args.task_type == 'multiclass' and y.ndim > 1 and y.shape[-1] == 1: y_target = y.long().squeeze(-1)
                    elif args.task_type == 'multiclass': y_target = y.long()
                    else:
                        y_target = y.float()
                        if y_target.ndim == 1: y_target = y_target.view(-1,1)
                        if output.ndim == 1 : output = output.view(-1,1)
                    loss = criterion(output, y_target)
                    # --- End Calculate Loss ---

                    # Add adversary loss to val_loss if desired (usually not for ES)
                    if n_gpu > 1: loss = loss.mean()
                    val_loss += loss.item()

        val_loss /= len(val_generator) if len(val_generator) > 0 else 1
        print(f'Epoch {epoch} - Val loss: {val_loss:.5f}', flush = True)

        # Early stopping check
        es(val_loss, checkpoints)
        if es.early_stop:
            print(f"Early stopping triggered after epoch {epoch}", flush=True)
            break

    # --- End of Training Loop for one predictor config ---
    print(f'Finished training for predictor config. Trained for {epoch} epochs.', flush=True)

    # Load best model based on early stopping
    print("Loading best model checkpoint based on validation loss...", flush=True)
    # Need to handle loading state dict potentially into DataParallel module
    best_predictor_state_dict = load_checkpoint(PREDICTOR_CHECKPOINT_PATH)
    if isinstance(predictor, nn.DataParallel):
        predictor.module.load_state_dict(best_predictor_state_dict)
    else:
        predictor.load_state_dict(best_predictor_state_dict)

    try: os.remove(PREDICTOR_CHECKPOINT_PATH)
    except OSError as e: print(f"Error removing predictor checkpoint: {e}")

    if not args.freeze_bert and os.path.exists(MODEL_CHECKPOINT_PATH):
        best_model_state_dict = load_checkpoint(MODEL_CHECKPOINT_PATH)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state_dict)
        else:
            model.load_state_dict(best_model_state_dict)
        try: os.remove(MODEL_CHECKPOINT_PATH)
        except OSError as e: print(f"Error removing model checkpoint: {e}")


    # --- Grid Search Evaluation (if applicable) ---
    if args.gridsearch_classifier:
        print("Evaluating predictor config on validation set for grid search...", flush=True)
        auprcs = []
        prediction_dict_val_gs, _, _ = evaluate_on_set(val_generator, predictor, emb_gen=args.freeze_bert)

        for c_val in c_grid:
            merged_preds_val = merge_preds(prediction_dict_val_gs, c_val)
            merged_preds_val_list = [merged_preds_val[str(i)] for i in actual_val.index if str(i) in merged_preds_val]
            actual_val_list = [actual_val.loc[int(i)] for i in actual_val.index if str(i) in merged_preds_val]

            if not merged_preds_val_list or not actual_val_list:
                 print(f"Warning: No matching predictions found for c_val={c_val}. Skipping AUPRC.", flush=True)
                 auprcs.append(0.0)
                 continue

            if args.task_type == 'binary':
                 actual_val_int = np.array(actual_val_list).astype(int)
                 # Ensure predictions are valid probabilities
                 valid_preds = np.clip(np.array(merged_preds_val_list), 0, 1)
                 if len(actual_val_int) != len(valid_preds):
                      print(f"Warning: Mismatch in length for AUPRC calc (actual:{len(actual_val_int)}, pred:{len(valid_preds)}). Skipping.")
                      auprcs.append(0.0)
                      continue
                 if len(np.unique(actual_val_int)) < 2:
                      print(f"Warning: Only one class present in actual labels for AUPRC calc. Skipping.")
                      auprcs.append(0.0) # AUPRC undefined for single class
                      continue
                 try:
                      auprc = average_precision_score(actual_val_int, valid_preds)
                      auprcs.append(auprc)
                 except ValueError as e:
                      print(f"Error calculating AUPRC for c={c_val}: {e}. Skipping.")
                      auprcs.append(0.0)

            else:
                 auprcs.append(0.0) # Placeholder for non-binary

        print("AUPRCs for c_grid:", auprcs, flush = True)
        print("c_grid:", c_grid, flush = True)
        idx_max = np.argmax(auprcs) if auprcs else -1

        if idx_max != -1 and auprcs[idx_max] > 0: # Ensure we found a valid AUPRC
            best_auprc_for_config = auprcs[idx_max]
            optimal_c_for_config = c_grid[idx_max]
            grid_auprcs.append(best_auprc_for_config)
            # Store state_dict on CPU
            if isinstance(predictor, nn.DataParallel):
                es_models.append(predictor.module.cpu().state_dict())
            else:
                 es_models.append(predictor.cpu().state_dict())
            optimal_cs.append(optimal_c_for_config)
            print(f'Predictor Config Val AUPRC: {best_auprc_for_config:.5f} (Optimal c: {optimal_c_for_config})', flush=True)
            # Move predictor back to original device if needed for next iteration
            predictor.to(device)
        else:
             print("Warning: Could not determine best AUPRC for this config.", flush=True)
             grid_auprcs.append(0.0)
             es_models.append(None) # Mark as failed
             optimal_cs.append(c_grid[0]) # Default


# --- Post-Training / Gridsearch Selection ---
if args.gridsearch_classifier:
    if not grid_auprcs or all(m is None for m in es_models):
        raise ValueError("Gridsearch completed but no valid models were found/saved.")
    print("Selecting best predictor configuration from grid search...", flush=True)
    print("Grid AUPRCs:", grid_auprcs)
    print("Optimal Cs per config:", optimal_cs)
    idx_max_grid = np.argmax(grid_auprcs)

    best_predictor_state_dict = es_models[idx_max_grid] # Already on CPU
    best_predictor_params = grid[idx_max_grid]
    print("Best Predictor Params:", best_predictor_params)

    # Re-initialize the best predictor architecture and load state dict
    predictor = Classifier(**best_predictor_params).to(device) # Move to device first
    predictor.load_state_dict(best_predictor_state_dict)
    if n_gpu > 1: # Re-wrap if needed AFTER loading state dict
        predictor = torch.nn.DataParallel(predictor)

    opt_c = optimal_cs[idx_max_grid]
    print(f"Best overall Val AUPRC: {grid_auprcs[idx_max_grid]:.5f} with c = {opt_c}", flush=True)
else:
    # If not grid searching, the single trained predictor is the final one
    # It should already be loaded with best weights from ES
    opt_c = 2.0 # Default value if not using gridsearch_c


# --- Final Evaluation on Val and Test Sets ---
print(f"\nEvaluating final model on validation set using c = {opt_c}...", flush=True)
# Ensure predictor is on the correct device and in eval mode
predictor.to(device).eval()
if not args.freeze_bert: model.to(device).eval()

prediction_dict_val, merged_preds_val, embs_val = evaluate_on_set(val_generator, predictor, emb_gen = args.freeze_bert, c_val = opt_c)

# --- Final Metric Calculation (Ensure alignment and validity) ---
# ... (Keep the metric calculation section, ensuring robustness) ...
actual_val_list = []
merged_preds_val_list = []
valid_indices = [idx for idx in actual_val.index if str(idx) in merged_preds_val]
if not valid_indices:
     print("Warning: No overlapping predictions found for validation set metrics.", flush=True)
else:
     actual_val_list = [actual_val.loc[int(i)] for i in valid_indices]
     merged_preds_val_list = [merged_preds_val[str(i)] for i in valid_indices]

     if args.task_type == 'binary':
         actual_val_int = np.array(actual_val_list).astype(int)
         preds_array = np.clip(np.array(merged_preds_val_list), 0, 1)
         if len(np.unique(actual_val_int)) < 2:
              print("Warning: Only one class present in validation labels. Cannot calculate ROC/AUPRC.")
              acc = accuracy_score(actual_val_int, preds_array.round())
              ll = log_loss(actual_val_int, np.clip(preds_array, 1e-7, 1 - 1e-7))
              print('Final Validation Metrics:')
              print(f'  Accuracy: {acc:.5f}')
              print(f'  Log Loss: {ll:.5f}')
              print(f'  AUROC: N/A')
              print(f'  AUPRC: N/A')
         else:
              acc = accuracy_score(actual_val_int, preds_array.round())
              auprc = average_precision_score(actual_val_int, preds_array)
              ll = log_loss(actual_val_int, np.clip(preds_array, 1e-7, 1 - 1e-7))
              roc = roc_auc_score(actual_val_int, preds_array)
              print('Final Validation Metrics:')
              print(f'  Accuracy: {acc:.5f}')
              print(f'  AUPRC: {auprc:.5f}')
              print(f'  Log Loss: {ll:.5f}')
              print(f'  AUROC: {roc:.5f}')
     elif args.task_type == 'regression':
          mse = mean_squared_error(actual_val_list, merged_preds_val_list)
          print('Final Validation Metrics:')
          print(f'  MSE: {mse:.5f}')
     elif args.task_type == 'multiclass':
          actual_val_int = np.array(actual_val_list).astype(int)
          # Ensure merged_preds_val_list contains class indices
          pred_indices = np.array(merged_preds_val_list)
          report = classification_report(actual_val_int, pred_indices)
          print('Final Validation Metrics:')
          print(report)


print(f"\nEvaluating final model on test set using c = {opt_c}...", flush=True)
prediction_dict_test, merged_preds_test, embs_test = evaluate_on_set(test_generator, predictor, emb_gen = args.freeze_bert,  c_val = opt_c)

if args.output_train_stats:
    print(f"\nEvaluating final model on training set using c = {opt_c}...", flush=True)
    prediction_dict_train, merged_preds_train, embs_train = evaluate_on_set(training_generator, predictor, emb_gen = args.freeze_bert, c_val = opt_c)
else:
    merged_preds_train, embs_train = {}, {}

# --- Save Final Artifacts ---
# Retrieve final predictor params (handle DataParallel)
final_predictor = predictor.module if isinstance(predictor, nn.DataParallel) else predictor
# Assuming the Classifier class stores its config like {'input_dim': ..., 'num_layers': ...}
# If not, save the best_predictor_params found during gridsearch or the initial grid[0]
final_predictor_params_to_save = best_predictor_params if args.gridsearch_classifier else grid[0]

print("\nSaving final predictor parameters...", flush=True)
# Ensure params are serializable
serializable_params = {k: v for k, v in final_predictor_params_to_save.items() if isinstance(v, (int, float, str, bool, list, dict))}
json.dump(serializable_params, open(os.path.join(args.output_dir, 'predictor_params.json'), 'w'))

print("Saving final predictor state dict...", flush=True)
torch.save(final_predictor.state_dict(), os.path.join(args.output_dir, 'predictor.pt'))

# Save model (if not frozen)
if not args.freeze_bert:
    print("Saving fine-tuned BERT model...", flush=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # Ensure config is saved correctly
    if hasattr(model_to_save, 'config'):
         model_to_save.config.to_json_file(output_config_file)
    else:
         print("Warning: Could not find config attribute on model to save.")
    tokenizer.save_vocabulary(args.output_dir)

# Save args
print("Saving script arguments...", flush=True)
# Convert Path objects or other non-serializable types if necessary
serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
json.dump(serializable_args, open(os.path.join(args.output_dir, 'argparse_args.json'), 'w'))

# Saves embeddings
if args.save_embs:
    print("Saving embeddings...", flush=True)
    embs = {**embs_val, **embs_test, **embs_train}
    pickle.dump(embs, open(os.path.join(args.output_dir, 'embs.pkl'), 'wb'))

# Save merged predictions
print("Saving merged predictions...", flush=True)
rough_preds = {**merged_preds_val, **merged_preds_test, **merged_preds_train}
pickle.dump(rough_preds, open(os.path.join(args.output_dir, 'preds.pkl'), 'wb'))

# Saves gridsearch info
if args.gridsearch_classifier:
    print("Saving gridsearch info...", flush=True)
    gs_info = {
        'grid_auprcs':grid_auprcs,
        'optimal_cs_per_config': optimal_cs,
        'best_overall_opt_c': opt_c
        }
    # Add best predictor params if available
    if 'best_predictor_params' in locals():
         gs_info['best_predictor_params'] = {k: v for k, v in best_predictor_params.items() if isinstance(v, (int, float, str, bool, list, dict))} # Ensure serializable
    pickle.dump(gs_info, open(os.path.join(args.output_dir, 'gs_info.pkl'), 'wb'))

print("\nScript finished successfully.", flush=True)