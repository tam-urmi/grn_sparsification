# Bio-GRN data

The former `biogrn_sparse` workspace has been merged into this project.
The TCGA/PANDA network inputs are now stored locally in
`data/raw/tcga_cancer/`; they are not included in the public Git repository.
Check the source, license, and redistribution terms for each dataset before
sharing or publishing it.

The YEASTRACT downloader is in `src/data/download_yeastract.py`. By default it
writes downloaded files into `data/raw/yeastract/`.
