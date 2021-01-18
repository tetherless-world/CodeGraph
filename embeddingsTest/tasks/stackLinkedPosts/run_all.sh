
python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json distilbert_para /dev/null > distilbert.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json xlm /dev/null > xlm.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json bert /dev/null > bert.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json msmacro /dev/null > msmarco.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json USE /dev/null > USE.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json bertoverflow /data/BERTOverflow/ > bertoverflow.results

python test_linked_posts.py /data/blanca/stackoverflow_data_linkedposts__testing.json finetuned /data/BERTOverflow-tuned/linked_posts/dccstor/m3/blanca/BERTOverflow/-2021-01-14_10-23-55/0_Transformer/ > tuned.results


