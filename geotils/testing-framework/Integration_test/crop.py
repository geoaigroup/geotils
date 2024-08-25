import geotils.data_processing.cropping as crop

"""outputs alot of files hence it is not called"""
class test_crop:
    Cropper = crop.CropGeoTiff()

    def test_crop_images(self):
        self.Cropper.crop_images(
            input_dir=r"geotils_testing\shapetest2\tiff\test2",
            output_dir=r"geotils_testing",
        )

    def test_crop_shapefiles(self):
        self.Cropper.crop_shapefiles(
            r"geotils_testing\noauxc",
            r"geotils_testing\shapetest2\shape",
            r"geotils_testing",
        )

    def test_crop_nongeoref_images_shapefile(self):
        self.Cropper.crop_nongeoref_images_shapefile(
            r"geotils_testing\noauxc\here",
            r"geotils_testing\shapetest2\shape",
            r"geotils_testing",
            r"geotils_testing",
        )
