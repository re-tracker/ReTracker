- add CHANGELOG;
- add three frame matching training backbone;
- add three frame loss;
- add `cross_attn_mode` of transformer.py in config;

- debug `updated_pos` in pips_refinement;

- remove temporal fusion module;

- remove DDPSampler

- remove 3 frame matching(and consistency check),
- change the coarse memory to pre-attention memory(backbone memory) ;
- polish code

- refactor training & solve the memory OOM bug
  - [x] dataset 
  - [x] trainval_inference
  - [x]plot function

- allowed change `backbone resolution` and `image resolution` in config;
- change default trainer config

- update requirements_a100.txt;
- increase validation plotting interval from 5 to 50;
- update run_a100.sh, selecting the personal path automatically;

- debug DDP error during sanity check(introduced when fixing memory OOM)
- remove default CUDA DEVICES selection

- change flyingthings sample rate;

- change a100_run.sh for better outputs structure;
- add recommit bash;

- adaptive link selection in 3 different developing platform; 
- update movi-e dataset;
- debug dataset length caculation

- **IMPORTANT** ADD support for BZ>1
  - debug loading bug: self.S, now dataset length align with the length of videos;
  - ADD new loading strategy!
    - set `batch_size=1` for pl(faked batchsize), while use `real_batch_size` for training; (avoid wrong training order for temporal data)
  - add multi-thread data loading;
- minor changes about env;
- recover DDPSampler;
- Fix TensorBoard evaluation plotting bug;

- add follow up strategies
- debug kubrics_512 datasets for training

- polish repo
- add loss display in bar
- add modelcheckpoint callback
- debug tensorboard PCK log missing

- debug PCK based ModelCheckpoint;
- add feature: elastic training;
- change IO mode

- change default evalutation settings in ant

- recover pretrained loftr model for better training

- remove SaveSharedCheckpointCallBack

- add use_dino config, and set default=false

- change dino input resolution from 512/8*14=896 to 448;
- change default training dataset;

- use 32x32 dino backbone;
- supervise dino branch separately;

- use DINO as coarse feature;
  - use_dino
  - upsample=True
  - w/ attention layer
- use CNN as mid and fine

- use non-interpolated 64x64 dino

### V2.0 ADD DINOV2 32x32 
step 1.
[x] add DinoV2 32x32 backbone on top of CNN backbone;
[x] polish corresponding codes;
[x] regress stage x3
[x] set supervise for 32x32 coarse input;

[x] final coarse loss = 
  classifier loss (loftr + dinov2) +
  regression loss 3 layer regression;
[x] debug and running
[x] add OcclusionCallback
[x] test running;
[x] remove 3 frame, add pairwise training;

step 2.
[x] running on Megadepth e.t.c. 
  [x] set new flag for loftr.py
  [x] set new dataloader;
  [x] set new trainer;
  [x] set new loss; 
[-] refactor confidence computation;
[x] set the init uncertainty of refine stage to zero;

[x] add dist plot mode for matching task
[x] add custom callback config

[x] add RoPE attention;
[x] add Flash Attention;

[x] add test code for matching task;

[x] test:
    - add detector-free(generate grid queries automatically) matching method;
    - remove dependency of depth during training;

[x] set validation as yfcc1500;
[x] add plotting for test;

[x] add uncertainty learning for matching task
[x] adjust val interval
[x] set uncertainty filter for test mode;

[x] minor plotting changes

P0
[] debug & train tracking method;

### V2.1 apply tricks from RoMA
[x] plot training PCK @1 @3 @5;
[x] remove regression supervision in dino level(detach dino refinement);
[x] use robust regression loss;
[x] separate uncertainty;
[x] pred conf for unmatchable points or ambiguous matches;
[x] only take certain point into account during bp;
[x] use dense fine supervision(>>1000)
[x] train on hard list;

### V2.2 apply RoMA-like Encoder-Decoder structure for coarse matching prediction
[x] replace mlp by tiny transfomer block
[x] add backbone, change loss, e.t.c.
[x] add pcache training for acc

### V2.3 change correlation:
[x] put image pairs to images all together
[x] add coarse pe;
[x] set unmatchable keypoints to dustbin(unbalanced label), and use focal loss;
[x] balanced sampling for training:
  - select 2000 matches(valid first for training)
[x] fix plotting UI;

### V2.4 
[x] add dino_cross_attn: change prior from single image to pairwise prior
[x] separate log and evaluation function;
[x] remove false queries from plotting during training
[x] remove unreliable matches during evaluation
[x] change default training plotting to PCK;
[x] add correlation relation;


### V2.5 self-affinity encodeded feature
[x] remove pretrained resnet backbone;
[x] change default Gray data to RGB data;
[x] set toggle of trainable residual in dino 


## V2.6 dino coarse decoder
[x] add coarse feature(8x) for decoding
[x] change supervision of certainty in coarse and fine
[x] add logger filter and dump results.
[x] add self-correlation(as roma does)
[x] change kernel function to 0-1

## V2.7
[x] dino coarse stage provide 1024=32x32 patches for 512px images;
[x] coarse supervision caculate loss for 32x32 position, fine: supervise sampled 2000 kpts from 64x64 position;
[x] change the logic of decision: if a keypoint is matchable: > bin score
[x] change Transformer to meta version;
[x] debug location: j provide 1024 coordinate version


## V2.8
[x] change refinement stage from cascade N stage to parallel N stage iterative update;
- local iterative refinement based on condition feature;
[x] add FINE_LOSS_THRESHOLD params
[x] update mlp mixer
[x] trim the refinement stage; combine coarse & fine refinement together;
  - cost map based refinement, based on PIPS;
[x] replace mlp by transformer
[x] change refinement windows_size_f;
[x] add daemon for elastic training & fix bug
[x] update coarse stage: set 1024 to 4096 supervision; update default M to 2000;
[] set VGG as default CNN selection;

## V2.9 IMPORTANT UPDATE
[x] debug tracking training pipeline for newest matching model;


## V2.9 IMPORTANT UPDATE
[x] warp matches of coarse dino for refinement;
[x] use coarse pips only
[x] transfer all matches to relative (-1,1)
[x] [trainable] [bad perfromance] point to patch PIPS refinement;


## V3.0
[x] add loftr block after warp;
[] predict matches as fast as possible;
P0: predict occlusion as precise as possible;
[] predict occlusion in new way: sample matches 
[x] change default wz=11, pretrained1.ckpt 
[x] change fine loss thresh to 16
[x] remove(comment) warp part for faster training
[x] add 4iter refinement for better performance
[x] debug savesharedcheckpoint callback

## V3.1
[x] implementation of RoMA refinement;
[x] fix savesharedcheckpoint callback bug;+2
[x] fix tracking config;

## V3.2
[x] add temporal layer for dino feature;
[x] set VGG19 as default CNN backbone, fixbug+1

[x] set pure dino feature as memory;
[x] set resnet as default backbone, add backbone selection config;

[x] onekey evaluation code [draft];
others:
[x] update daemon for elastic training for tracking;
[x] update default tracking strategies config + 1
[x] fix daemon bug;
[x] debug evaluation bash; + 1
[x] add coarse_flow_conf;


## V3.3
[x] add encoding layer for memory, so that all history features can be used in bp; [debug+1]
[x] back to 4096 tokens [debug+1]
[x] change default tracking config

[x] add temporal attention;
[x] resize DINO feature to 892, so that we have larger 64x64 feature map;
[x] set reasonable thresh for occlusion prediction and memory;

[x] fixbug: save interpolated precise dino feature(not 1/32 or 1/64) for anchors to keep the memory consistency;
[x] fixbug +2
[x] remove temporal fusion part;
[] remove original temporal attention layer, add Kernel temporal action for DINO;

[x] build longer flyingthings dataset, in line with kubrics[frames];
[x] polish lightning dataset code;
[x] change queries attention layer structure;
[x] add memory dropout/replace during training

[x] polish: memory_management code, dino_encoder Class config;
[x] remove 2D position encoding for dino features as well;
[x] change default training/testing config
[x] fixbug of aforemetioned features +2

[x] add temporal attention layer for refinement;
[x] set replace rate = 0, remove position encoding for conv8x +1
[x] add symmetric_index_v2 for flyingthings dataset +1
[x] add temporal_query_attn for each level's training

[x] use pretrained model; add temporal attention layer;

[x] update memorymanager; we can toggle detach grad or not +1

## V4.0 
[x] replace flow refinement by PIPS refinement;
[x] keep precise interpolated queries features; fixbug +1 ;
[x] add coarse 8x supervision for pips features

[x] add loftr self_cross part for refinement
[x] align matching task config with tracking task

[x] change default loss type to huber
[x] add coarse loftr encoder for cnn

[x] remove coarse loftr encoder, replace dino decoder with [coarse + dino] decoder;
[x] add negative supervision for dino decoder based method on megadepth dataset; 

[x] replace [coarse + dino] decoder with coarse;
[x] detach feat_c and queries feat from refinement stage(train coarse correspondences only);
[x] add fine pips;

[x] align tracking with matching;
[x] recover [key-component] [coarse + dino] decoder; +1

[x] back to best state
[x] add accurate queries; set queries to 400 for matching and tracking; [fix bug+1]
[x] set 1024 anchor queries instead of 4096 for faster training;
[x] change temporal attention type to linear; change pretrained ckpt; +1
[x] add pointodyssey dataset; add pcache support; +2
[x] add virtual keypoints for evaluation when max keypoints<400; +1; change the implementation.
[x] polish code; fix training/inference toggle bug +1, remove some comments codes
[x] remove coarse CNN features in roma backbone;
[x] misc changes: add new pe func, add vgg backbone func;
[x] add config: `train_fine` to skip fine stage; +1

[x] update training config: change default queries num;
[x] polish code +1; change pos encoding scheme;
[x] refactor matching pipeline in main;
[] add training scheme: return a bunch of predictions for multiple frames;
    - [x] datasets return image pairs/ image sequences / video sequences;
    - [x] adjust lightning pipeline for image_matching, video_matching task;
    - [x] add support for video matching task;
    - IO: video input -> separate pairwise matching

9.3
[x] fix introduced bug by last commit;
[x] add multi-frame S>8 support
    - [x] init coarse results by previous predictions;
    - [x] first queries scheme;
[x] custom sliding-window size;

9.6
[x] fix detach bug
[x] replace unfold + j_idx crop by precise grid_sample crop
[x] fix bug, using updated_feat for (queries * correlation);

9..7
[x] EMA
[x] detach half, training last half;
[x] learn to regress offset;

9.8
[x] add spattemporal attn layer +1
[x] update padding strategy;
[x] add config yaml
[x] replace offset with new position
[x] normalize conv feat;
[x] update movi-e dataset

9.10
[x] polish by pylance;
[x] refactoring PIPS module
[x] remove backbone feature norm & PE (add dim**0.5 norm)
[x] fix spat/temporal attn bug;
[x] polish code, ignore some pylance warning
[x] add regress loss thresh for ignoring oow;

[x] add updated feat to mlp output; combine output with queries feat;
[ing] fix failure sampling bug; resampling till success;
[-] come up with new branch which merge two branches together 
[-] add occlusion supervision;
[-] (optional) add updated_feat for refinement stage;
[] online training & inference:


9.12
[x] average loss with ant training version;

9.13
[x] update one-key evaluation scripts;
[x] polish code; update video logging code;
[x] test mix-precision training; update training config;
[x] add fine level

9.14
[x] add 3 layers refinement(dino/coarse/fine); 
9.16
[x] remove clone in normalize and denormalize; 
[x] featmap_pym_i[:,f].detach();

9.18
[x] change correlation patch method(align with cotracker)
[x] add multilayer supervision & loss
[x] recover to delta output

[x] [polish] add comments for functions;
[x] change pe & xy_to_ids function;

[ing] P0:add occlusion estimation loss and prediction for tracking task;
[ing] P1:add 4x/2x refinement layer for refinement;
[ing] P1:eval generated warped images and add aug choices;

[ing] P4:build pointodyssey dataset trainer;

9.22
[x] remove debug output
[x] align image matching training;
[x] add cotracker-Tracer API;

9.23
[x] add certainty for matching training 
[x] refactor evaluation and plotting, take certainty scores into account;
[x] add tmp setup.py for others
[x] plot settings/ build_retracker function
[x] check video_matching pipeline;
[x] change default pointodyssey dataset settings
[x] polish DINO backbone; add more configs;
[x] update all yaml;

[x] update certainty caculation method;
[x] refactor sampling function, raise more valid matches;
[x] change the ignore index of multi-focal-loss from 0 to 4096
[x] polish plotting code for video tracking
[x] debug the class of unmatched keypoints to bin 4096;

[x] add refinement block for matching task;
[x] debug corr of refinement stage, apply patch based refinement;
[x] fix supervision, suppport 1vN GT supervision for matching & tracking;
[x] add temporal attn

[x] fix no config error;
[x] use coarse pred_certainty loss as instruction;
[x] refactor dataloader, now all inputs are videos rather than frames;
[x] skip validation fix;

[x] add causal state for fine tracking;
[x] add 4d correlation;
[x] set default corr_rad = 3, iter_num = 4
[x] change training status, debug

[x] recover 2D position encoding for dino features(important) IMPORTANT
[x] add kill_process helper

[x] add support for extra configs;
[x] summary pips_refinement parameters
[x] polish code;
[x] init patch_attn
[x] patch_attn 1021version
[x] align patch_attn; add temporal_embed;

[x] change huber loss threshold;
[x] add adaptive selection from [prev, match pred]
[x] change training keypoints num to 512;
[x] remove spatial attn;
[x] reset training settings;
[x] add eval mode;
[x] add occlusion & confidence learining;
[x] update retracker engine;
[x] update evaluate code;
[x] reweight loss, add multiscale feature for mlp_input;
[x] debug
[x] add localtrack pytorch
[x] comment localtrack related codes during inference

[x] remove corr4d_i for each layer, add all 4 layers for each iteration
[x] use follow up strategy during training steps;
[x] add epe and pck plotting
[x] add cmdtop;
[x] add coarse features;
[x] fix movi-e dataset sampling bugs(valids=valids&vis to valids=valids);
[x] add lrscheduler
[x] widen hidden layers from 256 to 384; causal context from 4 to 8;
[x] change the way of collecting causal context;
[x] add RoPE for PIPS
[x] update to troma version
[x] update troma;
[x] update randomwalk troma;
[x] use 3x refinement for training efficiency;
[x] update loss weights, update nan callback[+1];
[x] dump some changes
[x] update eval scripts
[x] remove patch SinePE
[x] back to troma version
[x] adjust channels;
[x] add dino layer;
[x] add temporal layers;
[x] remove inv;
[x] weighted loss
[x] square mem
[x] hard sampling[fixbug]
[x] remove default scheduler, add dump state callback
[x] switch to automatic optimization to prevent nan/inf;
[x] set 32 precision
[x] add multi-level certainty/occlusion supervision;
[x] back to randomwalk;
[x] add global PE for passing information between levels;
[x] add new pe
[x] back to troma
[x] back to randomwalk
[x] change dropout
[x] add jitter
[x] add panning_movie dataset
[x] add adaptive_12 and update related memory settings
[x] change jittering setting
[x] debug occlusion, debug panning movi-e dataset;
[x] updated loss
[x] use updated feat, rather than the feat itself;
[x] add demo;
[x] change training dataset settings
[x] add evaluation scripts for kubrics and rgbstacking;
[x] mix all tokens for romatransformer decoder:
[x] add support for kubrics/panning_movie dataset switch
[x] fix plotting bugs
[x] change 2 stage mode to dino 16x + cnn 2x;


# V_Paper 5.1
[x] align with online sota
[x] polish and add multilvl tokens function;
[x] imp multilvl tokens;
[x] add prior tokens for next level;
[x] add tanh and independent troma_c

[x] add multiscale reception field
[x] replace with dino adaptor bkbn
[x] to full dino adaptor based backbone
[x] use 4x 8x 16x feat for corr_block
[x] back to 2x 8x CNN & pretrained 16x dino
[x] update evaluator
[x] align config with current sota;
[x] fixbug
[x] change default backbone to pretrained VGG16_BN;
[x] add temporal RoPE
[x] remove affinity_matrix_0i
[x] add coarse feature rather than pooling;
[x] fix affinity matrix bug;
[x] use resnetfpn and allow grad
[x] add feature level spat-temp attention
[x] add self-gated sumup;
[x] fix evaluator
[x] fix prior patch_random_i; 
[x] update to remote best config(retracker_0130v5);
[x] update matching block
[x] add Robotap evaluation;
[x] align matching and tracking config and model;
[x] update video matching config;
[x] update config;
[x] fix OOM bug
[x] add temporal attn layer;

[x] update to latest evaluation version;
[x] add scannet++ training code;
[x] new training scheme: allowed memory BP for video matching;
[x] add k_epic datasets [fix]

[x] update evaluation codes;
[x] adapt multi-node training code for tapvid;
[x] self updated temporal attn to self gated attention;

[x] updt visualizer: add hide_occ_points feature;
[x] updt memorymanager: add 15 frames;
[x] add AUG mode for teaser: rotate & eval;

[x] add tools for convert ckpt;
[x] change loftr default parameters, align with best evaluation settings; +1

[x] update demo code;
[x] update visualzier
[x] change default loftr config;
[x] update visualizer to @yiqing version;
[x] update evaluation & visualization codes;
[x] polish codes
[x] update inference code;


## alpha 0.2
- change the location of cli;
- refactor repo;
- polish loftr.py  x1
- remove keywords: loftr;

- fix setup and tracking demo;
- update rich_utils;
- polish code;
- polish
- polish codes;

- FEAT: allowed matching evaluation with BF16

------ TODO@future
[] mix all datasets for training;

# change config
# polish code

## alpha v0.9
1. set adaptive WORKDIR for dino backbone;
2. polish retracker.py;
3. update engine.py;
4. loss: adapt flow loss weights;
5. loss: set wo_safe_mask = False for fine loss caculation;