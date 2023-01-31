!/bin/bash
source ~/.bashrc
echo "activate environemnt !"
conda activate fairseq
echo "start running job !"



#example for running on atis_logic
fairseq-train parsing_data/atis_logic-bin --task semantic_parsing --dataset-impl raw \
											--arch transformer_parsing --share-decoder-input-output-embed \
											--optimizer adam --clip-norm 1.0 --lr 1e-3 --warmup-updates 4000 \
											--criterion cross_entropy --batch-size 60 --sentence-avg --max-update 20000 \
											--log-format simple --save-dir atis_logic_baseline_transformer \
											--no-epoch-checkpoints \
										    
fairseq-train parsing_data/atis_logic-bin --task semantic_parsing --dataset-impl raw \
											--arch lstm_parsing --share-decoder-input-output-embed \
											--optimizer adam --clip-norm 1.0 --lr 1e-3 --warmup-updates 4000 \
											--criterion cross_entropy --batch-size 60 --sentence-avg --max-update 20000 \
											--log-format simple --save-dir atis_logic_baseline_lstm \
											--no-epoch-checkpoints \
									


fairseq-train parsing_data/atis_logic-bin --task semantic_parsing_tag -s word -t predicate -g tag \
										--data-name atis_logic --tag-input \
										--dataset-impl raw --arch transformer_tag_parsing \
										--share-decoder-input-output-embed --optimizer adam \
										--clip-norm 1.0 --lr 1e-3 \
										--criterion label_smoothed_cross_entropy --batch-size 60 \
										--warmup-updates 4000 --sentence-avg --max-update 20000 \
										--log-format simple --save-dir atis_logic_transformer_tag_v1 \
										--save-interval 10


fairseq-train parsing_data/atis_logic-bin --task semantic_parsing_tag -s word -t predicate -g tag \
										--data-name atis_logic --tag-input \
										--dataset-impl raw --arch lstm_tag_parsing \
										--share-decoder-input-output-embed --optimizer adam \
										--clip-norm 1.0 --lr 1e-3 \
										--criterion label_smoothed_cross_entropy --batch-size 60 \
										--warmup-updates 4000 --sentence-avg --max-update 20000 \
										--log-format simple --save-dir atis_logic_lstm_tag_v1 \
										--save-interval 10

