import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from torch.autograd import Variable
    
# ============================================================================
# Mapping module
# This class is a linear mapping to image space. rho(.)
class LinearMapping(torch.nn.Module):

    def __init__(self, image_embed_dim =512):
        """
        A function that takes in two image embeddings and maps them to a single image embedding.
        
        :param image_embed_dim: The dimension of the image embedding, defaults to 512 (optional)
        """
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        """
        It takes the input x and passes it through the mapping function.
        
        :param x: the input data
        :return: The output of the mapping function.
        """
        theta_linear = self.mapping(x[0])
        return theta_linear

# This class is a convolutional mapping to image space. 
# The input to this class is a list of tensors. The first tensor is the image embedding, and the rest
# are the features extracted from the image. 
# The output is a tensor of shape (batch_size, image_embed_dim). 
# The class has two sub-modules: 
# 1. mapping: This is a fully connected neural network that takes in the concatenated features and
# outputs a tensor of shape (batch_size, image_embed_dim). 
# 2. conv: This is a convolutional neural network that takes in the concatenated features and outputs
# a tensor of shape (batch_size, 64, 16). 
# The output of the convolutional neural network is then reshaped to (batch_size, 1024) and
# concatenated with the output of the mapping module.
class ConvMapping(torch.nn.Module):

    def __init__(self, image_embed_dim =512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(16)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1024))
        theta_conv = self.mapping(final_vec)
        return theta_conv

# ============================================================================
# Complex projection module
class ComplexProjectionModule(torch.nn.Module):

    def __init__(self, image_embed_dim = 512, text_embed_dim = 768):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score

# ============================================================================
# ComposeAE model
class ComposeAE(torch.nn.Module):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert):
        super().__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

        img_model = torchvision.models.resnet18(pretrained=True)

        class GlobalAvgPool2d(torch.nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, image_embed_dim))
        self.img_model = img_model

        # text model
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab = text_query,
            word_embed_dim = text_embed_dim,
            lstm_hidden_dim = text_embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(),
            ConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim)
        )

    # Extract feature
    def extract_img_feature(self, imgs):
        """
        It takes a list of images and returns a list of features extracted from those images
        
        :param imgs: a batch of images
        :return: The output of the img_model
        """
        return self.img_model(imgs)

    def extract_text_feature(self, text_query, use_bert):
        """
        If use_bert is True, then use the BERT model to encode the text query, otherwise use the text
        model to encode the text query
        
        :param text_query: the text query that we want to embed
        :param use_bert: whether to use BERT or not
        :return: The text features are being returned.
        """
        if use_bert:
            text_features = bc.encode(text_query)
            return torch.from_numpy(text_features).cuda()
        return self.text_model(text_query)

    # Loss
    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)

    def compute_loss(self, imgs_query, text_query, imgs_target, soft_triplet_loss=True):
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        composed_source_image = self.normalization_layer(dct_with_representations["repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm)
        assert (composed_source_image.shape[0] == target_img_features.shape[0] and
                composed_source_image.shape[1] == target_img_features.shape[1])
        # Get Rot_Sym_Loss
        CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
        conjugate_representations = self.compose_img_text_features(target_img_features_non_norm, dct_with_representations["text_features"], CONJUGATE)
        composed_target_image = self.normalization_layer(conjugate_representations["repres"])
        source_img_features = self.normalization_layer(dct_with_representations["img_features"]) #img1
        if soft_triplet_loss:
            dct_with_representations ["rot_sym_loss"]= \
                self.compute_soft_triplet_loss_(composed_target_image,source_img_features)
        else:
            dct_with_representations ["rot_sym_loss"]= \
                self.compute_batch_based_classification_loss_(composed_target_image,
                                                            source_img_features)

        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(composed_source_image,
                                                   target_img_features), dct_with_representations
        else:
            return self.compute_batch_based_classification_loss_(composed_source_image,
                                                                 target_img_features), dct_with_representations

    # Compose
    def compose_img_text_features(self, img_features, text_features, CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False)):
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations    

    def compose_img_text(self, imgs, text_query):
        """
        It takes in a list of images and a text query, and returns a list of image-text features
        
        :param imgs: a list of images
        :param text_query: a list of strings, each string is a query
        :return: the composed image and text features.
        """
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)
