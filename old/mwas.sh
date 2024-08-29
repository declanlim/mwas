# remove the folder from mnt if it exists
if [ -d /mnt/mwas ]; then
sudo umount -f /mnt/mwas
sudo rmdir /mnt/mwas
fi
# remove this (if it exists)

echo "Creating and mounting tmpfs folder"
# create and mount the tmpfs folder to keep files in RAM

sudo mkdir /mnt/mwas

# rnalab-mwas instance has 128GB RAM, mwas-setup folder 88.1GB
sudo mount -o size=95G -t tmpfs mwas /mnt/mwas # SET THE SIZE AS A PARAMETER
echo "Mounted tmpfs folder"

echo "Syncing MWAS setup files"
cd /mnt/mwas
aws s3 sync s3://serratus-biosamples/mwas_setup/ . --size-only
echo "Sync complete"

cd ~/mwas
split -l 55 -d --additional-suffix=.txt family_groups_all.txt family_group_

ls family_groups/ | parallel -j 40 --jl mwas_rfam.log "python mwas_rfam.py"
