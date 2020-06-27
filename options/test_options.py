from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--test_phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50000, help='how many test images to run')
        parser.add_argument('--input', type=str, default='examples/photo_examples/Scarlett_Johansson_P00002.jpg', help='input image path')
        parser.add_argument('--shape', type=str, default=None, help='shape image name')
        parser.add_argument('--style_path', type=str, default='examples/style_imgs/Angela_Merkel_C00012.jpg', help='style image path')
        parser.add_argument('--parsing_model', type=str, default='checkpoints/parsing.pth', help='parsing model path')
        parser.add_argument('--retrieval_model', type=str, default='checkpoints/retrieval.pth.tar', help='retrieval model path')
        parser.add_argument('--style_decoder_model', type=str, default='checkpoints/style_decoder.pth.tar', help='style decoder model path')
        parser.add_argument('--style_encoder_model', type=str, default='checkpoints/vgg_normalised.pth', help='style encoder model path')

        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
