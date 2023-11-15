import math
import os
import argparse
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from support import RatingDataset, read_img_feature, read_genres
from tqdm import tqdm
import pandas as pd
import time
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# torch.cuda.set_device(1)

# CCFCRec


class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()
        # In the attribute embedding table, the last dimension is used as padding with 0,
        # if an attribute of some item is missing, fill the corresponding position with 0,
        # the corresponding index is 18

        self.args = args
        self.attr_matrix = nn.Embedding(args.attr_num + 1, args.attr_present_dim, padding_idx=args.attr_num)
        # Define the attribute attention layer
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim))
        # Control the activation function of the entire model
        self.h = nn.LeakyReLU()
        # self.h = nn.ReLU()
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.softmax = torch.nn.Softmax(dim=0)
        # The mapping matrix of the image
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()
        # User and item embedding layers, can be initialized with pre-training
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
        # Define the generation layer, combine the information of (q_v_a, u) to generate q_v_c,
        # generating item embedding that contains collaborative information
        self.gen_layer1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # Parameter initialization
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.image_projection)
        # Initialize the generation layer
        # Initialize the user, item embedding layers, initialize if there is no pre-training
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, attribute, image_feature, user_id):
        attr_present = self.attr_matrix(attribute)
        attr_tmp1 = self.h(torch.matmul(attr_present, self.attr_W1.T) + self.attr_b1)
        attr_attention_b = self.softmax(torch.matmul(attr_tmp1, self.attr_W2))
        z_v = torch.matmul(attr_attention_b.transpose(1, 2), attr_present).squeeze()  # z_v is the vector of attributes after attention-weighted fusion
        p_v = torch.matmul(image_feature, self.image_projection)  # The image embedding vector of the item
        q_v_a = torch.cat((z_v, p_v), dim=1)
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c


def train(model, train_loader, optimizer, valida, args):
    print("model start train!")
    model_save_dir = 'result/' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    test_save_path = model_save_dir + "/result.csv"
    os.makedirs(model_save_dir)
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # Write hyperparameters
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(test_save_path, 'a+') as f:
        f.write("loss,contrast_loss,self_contrast_loss,p@5,p@10,p@20,ndcg@5,ndcg@10,ndcg@20\n")
    for i_epoch in tqdm(range(args.epoch)):
        i_batch = 0
        batch_time = time.time()
        for user, item, item_genres, item_img_feature, neg_user, positive_item_list, negative_item_list, self_neg_list in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)
            q_v_c = model(item_genres, item_img_feature, user)
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
            # Calculate contrastive loss
            positive_item_emb = model.item_embedding[positive_item_list]
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10
            # Calculate the negative cases
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]  # Calculate the mean for the same batch
            # self contrast
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2)/(
                args.tau*torch.norm(q_v_c_unsqueeze, dim=2)*torch.norm(self_neg_item_emb, dim=2))
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                    args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(self_pos_contrast_exp/(self_pos_contrast_exp+self_neg_contrast_sum))
            self_contrast_sum = torch.sum(self_contrast_val)
            # L_rank z_v & u
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).sum()
            # L_rank q_v & u
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()
            total_loss = args.lambda1*(contrast_sum+self_contrast_sum)+(1-args.lambda1)*(y_ukv+y_ukv2)
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            wandb.log(
                {'loss': y_ukv + y_ukv2, 'i_batch': i_batch, #batch_time: time.time() - batch_time,
                 'total_loss': total_loss.item(), 'contrast_loss': contrast_sum.item(),
                 'self_contrast_loss': self_contrast_sum.item()}
            )
            if i_batch % args.save_batch_time == 0:
                model.eval()
                print("[{},/13931603]total_loss:,{},{},s".format(i_batch*1024, total_loss.item(), int(time.time()-batch_time)))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                with open(test_save_path, 'a+') as f:
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(y_ukv+y_ukv2, contrast_sum, self_contrast_sum,
                                                                  hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
                wandb.log(
                    {'i_batch': i_batch, 'batch_time': time.time() - batch_time, 'p@5': hr_5, 'p@10': hr_10,
                     'p@20': hr_20, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10, 'ndcg@20': ndcg_20}
                )

                # Save the model
                batch_time = time.time()
                model_save_path = model_save_dir + '/epoch_' + str(i_epoch) + "batch_" + str(i_batch) + ".pt"
                torch.save(model.state_dict(), model_save_path)
                wandb.save(model_save_path, policy='now')


if __name__ == '__main__':
    # Argument parser
    args = get_args()
    wandb.init(project='ccfc', config=args)
    wandb.run.name = 'ML-20M b{} l{} -- {}' \
        .format(args.batch_size, args.learning_rate,
                wandb.run.id)
    wandb.run.save()

    # Extract the original id of the user: a dictionary for serialized id
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "./data/train_rating.csv"
    valid_path = './data/validate_rating.csv'
    train_df = pd.read_csv(train_path, dtype={'userId': int, 'movieId': int, 'neg_user_id': int})
    total_user_set = train_df['userId']
    user_serialize_dict = serialize_user(total_user_set)
    img_feature = read_img_feature('./data/img_feature.csv')
    movie_onehot = read_genres('./data/movies_onehot.csv')
    contrast_coll_path = "./data/user_positive_movie_48.csv"
    contrast_self_path = './data/self_contrast_48.csv'
    dataSet = RatingDataset(train_df, contrast_coll_path, img_feature, movie_onehot, user_serialize_dict,
                            contrast_self_path, args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("Model hyperparameters:", args_tostring(args))
    myModel = CCFCRec(args)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(validate_csv=valid_path, user_serialize_dict=user_serialize_dict, img=img_feature,
                         genres=movie_onehot)
    train(myModel, train_loader, optimizer, validator, args)

