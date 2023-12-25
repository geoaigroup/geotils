from pre_processing import LargeTiffLoader

largetile=LargeTiffLoader()
largetile.load_index("data_processing/images","data_processing","tif",0,0,1024,1024)