#!/bin/bash
./deploy_pre.sh
./deploy.sh
echo "for copy pasting to the lambda console:\n(deppack goes in mwas, mwas_large and mwas_small, deppack_pre goes in mwas_pre)"
echo "s3://serratus-biosamples/mwas_lambda_zips/deppack.zip"
echo "s3://serratus-biosamples/mwas_lambda_zips/deppack_pre.zip"
