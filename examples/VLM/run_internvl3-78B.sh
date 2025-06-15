export LMUData=dataset
export  Intern_API_KEY=$Your_key1 # for evaluated model


#********#
python -u run.py --data PhyX_mini_OE \
    --model InternVL3-78B-API \
    --judge-args '{"valid_type": "STR"}'


#********#
export SiliconFlow_API_KEY=$Your_key # for judger
python -u run.py --data PhyX_mini_OE \
    --model InternVL3-78B-API \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


#********#
export Deepseek_API=$Your_key # for judger
python -u run.py --data PhyX_mini_OE \
    --model InternVL3-78B-API \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'
