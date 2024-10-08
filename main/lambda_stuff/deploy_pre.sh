cd ../code
cp mwas_main.py ../lambda_stuff/lam_pack_pre/mwas_main.py
cp mwas_functions_for_preprocessing.py ../lambda_stuff/lam_pack_pre/mwas_functions_for_preprocessing.py
cd ../lambda_stuff
rm deppack_pre.zip
cd lam_pack_pre
zip -r ../deppack_pre.zip .
cd ..
aws s3 cp deppack_pre.zip s3://serratus-biosamples/mwas_lambda_zips/deppack_pre.zip
