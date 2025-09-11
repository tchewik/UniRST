local batch_size = std.extVar("batch_size");
local combine_batches = std.extVar("combine_batches");
local corpora = std.extVar("corpora");
local data_augmentation = std.extVar("data_augmentation");
local cuda_device = std.parseInt(std.extVar("cuda_device"));
local discriminator_warmup = std.extVar("discriminator_warmup");
local dwa_bs = std.extVar("dwa_bs");
local edu_encoding_kind = std.extVar("edu_encoding_kind");
local du_encoding_kind = std.extVar("du_encoding_kind");
local emb_size = std.extVar("emb_size");
local freeze_first_n = std.extVar("freeze_first_n");
local grad_clipping_value = std.extVar("grad_clipping_value");
local hidden_size = std.extVar("hidden_size");
local if_edu_start_loss = std.extVar("if_edu_start_loss");
local lr = std.extVar("lr");
local lstm_bidirectional = std.extVar("lstm_bidirectional");
local rel_classification_kind = std.extVar("rel_classification_kind");
local run_name = std.extVar("run_name");
local save_path = std.extVar("save_path");
local seed = std.extVar("seed");
local segmenter_dropout = std.extVar("segmenter_dropout");
local segmenter_hidden_dim = std.extVar("segmenter_hidden_dim");
local segmenter_separated = std.extVar("segmenter_separated");
local segmenter_type = std.extVar("segmenter_type");
local token_bilstm_hidden = std.extVar("token_bilstm_hidden");
local use_union_relations = std.extVar("use_union_relations");
local transformer_name = std.extVar("transformer_name");
local use_crf = std.extVar("use_crf");
local use_discriminator = std.extVar("use_discriminator");
local use_log_crf = std.extVar("use_log_crf");
local window_size = std.extVar("window_size");
local window_padding = std.extVar("window_padding");
local epochs = std.extVar("epochs");
local patience = std.extVar("patience");

{
    "data": {
            "corpora": corpora,
            "data_augmentation": data_augmentation
    },
    "model": {
        "transformer": {
            "model_name": transformer_name,
            "emb_size": emb_size,
            "normalize": true,
            "freeze_first_n": freeze_first_n,
            "window_size": window_size,
            "window_padding": window_padding
        },
        "segmenter": {
            "type": segmenter_type,
            "use_crf": use_crf,
            "use_log_crf": use_log_crf,
            "hidden_dim": segmenter_hidden_dim,
            "lstm_dropout": 0.2,
            "lstm_num_layers": 1,
            "if_edu_start_loss": if_edu_start_loss,
            "lstm_bidirectional": lstm_bidirectional,
            "separated": segmenter_separated,
        },
        "parser_type": "top-down",
        "hidden_size": hidden_size,
        "edu_encoding_kind": edu_encoding_kind,
        "du_encoding_kind": du_encoding_kind,
        "rel_classification_kind": rel_classification_kind,
        "token_bilstm_hidden": token_bilstm_hidden,
        "use_union_relations": use_union_relations,
        "use_discriminator": use_discriminator,
        "discriminator_warmup": discriminator_warmup,
        "dwa_bs": dwa_bs,
    },
    "trainer": {
            "tracker": "clearml",
            "lr": lr,
            "seed": seed,
            "epochs": epochs,
            "use_amp": false,
            "lr_decay": 0.95,
            "patience": patience,
            "project": "UniRST",
            "run_name": run_name,
            "eval_size": 200,
            "save_path": save_path,
            "batch_size": batch_size,
            "combine_batches": false,
            "weight_decay": 0.01,
            "lr_decay_epoch": 1,
            "lm_lr_mutliplier": 0.2,
            "grad_norm_value": 1.0,
            "grad_clipping_value": grad_clipping_value,
            "gpu": cuda_device,
    }
}