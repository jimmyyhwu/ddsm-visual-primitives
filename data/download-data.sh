echo Downloading data...
#wget -c http://data.csail.mit.edu/places/medical/data/ddsm_patches.tar.gz
#wget -c http://data.csail.mit.edu/places/medical/data/ddsm_labels.tar.gz
#wget -c http://data.csail.mit.edu/places/medical/data/ddsm_raw.tar.gz
#wget -c http://data.csail.mit.edu/places/medical/data/ddsm_masks.tar.gz
#wget -c http://data.csail.mit.edu/places/medical/data/ddsm_raw_image_lists.tar.gz
wget -c https://www.dropbox.com/s/xattw5gkfd72b1j/ddsm_patches.tar.gz
wget -c https://www.dropbox.com/s/766akqdm4jt0ksd/ddsm_labels.tar.gz
wget -c https://www.dropbox.com/s/33jx5t21avd0n5j/ddsm_raw.tar.gz
wget -c https://www.dropbox.com/s/x8l97k6rgihtg5x/ddsm_masks.tar.gz
#wget -c https://www.dropbox.com/s/pde2ohme5jxcuy4/ddsm_raw_image_lists.tar.gz

echo Unpacking data...
tar -xf ddsm_patches.tar.gz
tar -xf ddsm_labels.tar.gz
tar -xf ddsm_raw.tar.gz
tar -xf ddsm_masks.tar.gz
#tar -xf ddsm_raw_image_lists.tar.gz
