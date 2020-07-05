sam build -t "$2".yaml
sam package --output-template-file packaged.yaml --s3-bucket "$1"
sam deploy --confirm-changeset --template-file packaged.yaml --stack-name "$3" --capabilities CAPABILITY_IAM --parameter-overrides $(cat template_params.properties) --s3-bucket "$1"