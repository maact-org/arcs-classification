wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_pytorch_checkpoint.zip
wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt
unzip -a bert-base-portuguese-cased_pytorch_checkpoint.zip
mkdir "bert-base-portuguese-cased"
mv config.json bert-base-portuguese-cased/.
mv vocab.txt bert-base-portuguese-cased/.
mv pytorch_model.bin bert-base-portuguese-cased/.
rm bert-base-portuguese-cased_pytorch_checkpoint.zip