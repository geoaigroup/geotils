import unittest, torch, sys 
sys.path.append("../")
from models.Unet_W_Mods import UnetWMDecoder, UnetWM, DecoderBlock1x


class TestDecoderBlock1x(unittest.TestCase):
    
    def setUp(self):
        self.in_channels = 16
        self.skip_channels = 8
        self.out_channels = 32
        self.use_batchnorm = True
        self.attention_type = 'scse'
        self.decoder_block = DecoderBlock1x(
            self.in_channels,
            self.skip_channels,
            self.out_channels,
            self.use_batchnorm,
            self.attention_type
        )
    
    def test_forward_with_skip(self):
        x = torch.randn(1, self.in_channels, 64, 64)
        skip = torch.randn(1, self.skip_channels, 64, 64)
        output = self.decoder_block.forward(x, skip)
        self.assertEqual(output.shape, (1, self.out_channels, 64, 64))
    
    def test_forward_without_skip(self):
        x = torch.randn(1, self.in_channels + self.skip_channels, 64, 64)
        output = self.decoder_block.forward(x)
        self.assertEqual(output.shape, (1, self.out_channels, 64, 64))
    

class TestUnetWMDecoder(unittest.TestCase):
    
    def setUp(self):
        self.encoder_channels = [64, 128, 256, 512, 1024, 2048]
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.n_blocks = 5
        self.use_batchnorm = True
        self.attention_type = 'scse'
        self.center = False
        self.mod = None
        self.mod_configs = {}
        self.output_stride = 32
        self.in_size = (512, 512)
        self.decoder = UnetWMDecoder(
            self.encoder_channels,
            self.decoder_channels,
            self.n_blocks,
            self.use_batchnorm,
            self.attention_type,
            self.center,
            self.mod,
            self.mod_configs,
            self.output_stride,
            self.in_size
        )
    
    def test_forward_no_errors(self):
        features = [
            torch.randn(1, ch, 512 // (2 ** i), 512 // (2 ** i))
            for i, ch in enumerate(self.encoder_channels)
        ]
        try:
            output = self.decoder(*features)
        except Exception as e:
            self.fail(f"forward method raised an exception {e}")
    
    def test_forward_output_shape(self):
        features = [
            torch.randn(1, ch, 512 // (2 ** i), 512 // (2 ** i))
            for i, ch in enumerate(self.encoder_channels)
        ]
        output = self.decoder(*features)
        expected_shape = (1, self.decoder_channels[-1], 512, 512)
        self.assertEqual(output.shape, expected_shape)
    

class TestUnetWM(unittest.TestCase):
    def test_init(self):        
        model = UnetWM()        
        self.assertIsNotNone(model)

    def test_forward(self):        
        model = UnetWM()        
        x = torch.randn(2, 3, 384, 384)        
        output = model(x)        
        self.assertEqual(output.shape, (2, 1, 384, 384))

    def test_calling_forward(self):        
        model = UnetWM()        
        x = torch.randn(2, 3, 384, 384)        
        output = model.forward(x)        
        self.assertEqual(output.shape, (2, 1, 384, 384))


if __name__ == '__main__':
    unittest.main()
