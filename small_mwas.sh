# remove the folder from mnt if it exists
if [ -d /mnt/mwas ]; then
    sudo umount -f /mnt/mwas
    sudo rmdir /mnt/mwas
fi

echo "Creating and mounting tmpfs folder"
# create and mount the tmpfs folder to keep files in RAM
sudo mkdir /mnt/mwas
# Set an appropriate size for your machine, maybe around 4GB
sudo mount -o size=4G -t tmpfs mwas /mnt/mwas
echo "Mounted tmpfs folder"

# No need to sync MWAS setup files for this small test run

# Assuming you have a single family text file for testing
cd ~/mwas
# Skip splitting family_groups_all.txt and just use the single family file
python mwas_rfam.py family_group_01.txt
