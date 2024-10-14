batch_size = 1
modelname = "dtgroundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = False
use_transformer_ckpt = False
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

# train
aux_loss = True
freeze_all = True

# test
select_box_nums_for_evaluation = 200

#adapter
use_adapter = True
use_self_kd = False
encoder_gate_base_scale = 0.1
decoder_gate_base_scale = 0.1

# task agnostic
use_add_names = True

# cet
use_cet = False
cet_middle_dim = 1024
use_prompt_memory = False
cet_type = "Adapter"
use_prompt_memory_output = False

# zero loss
use_zero_inter_loss = False
loss_adapter_weight = 0.1

# moe
num_experts = 1
num_topk_experts = 1

# bert
use_bert_tuning = False

# linear probing
use_cls_linear = False

# prompt tuning
use_prompt_tuning = False