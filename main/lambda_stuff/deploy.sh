cd ../code
cp mwas_lambda.py ../lambda_stuff/lam_pack/mwas_lambda.py
cp mwas_functions.py ../lambda_stuff/lam_pack/mwas_functions.py
cd ../lambda_stuff
rm deppack.zip
cd lam_pack
zip -r ../deppack.zip .
cd ..
aws s3 cp deppack.zip s3://serratus-biosamples/mwas_lambda_zips/deppack.zip
