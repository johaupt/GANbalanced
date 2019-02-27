import wgan.data as data
from wgan.models_cat import Generator, Critic
from wgan.training import WGAN
from wgan.evaluation import discriminator_evaluation

from wgan.models_cat import make_GANbalancer


    X_train, X_test, y_train, y_test, idx_cont, idx_cat, cat_dict = load_DMC10("/Users/hauptjoh/Data/DMC10")
    
    X_train, X_test, y_train, y_test, Xy_gan, idx_cont2, idx_cat2, scaler =\
        data.prepare_data(X_train, y_train, X_test, y_test, 
                          idx_cont=idx_cont, idx_cat=idx_cat, 
                          cat_levels = [np.max(X_train[:,i])+1 for i in idx_cat])

def train_GAN(dataset, generator_input, generator_layers, critic_layers, cGAN, batch_size, epochs):

    emb_sizes = [int(min(10., np.ceil(x+1/2))) for x in dataset.cat_levels]
    
    generator, critic = make_GANbalancer(dataset, generator_input, generator_layers, critic_layers, emb_sizes, auxiliary=cGAN)
    
    batch_size = 64
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    #test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    # Initialize optimizers
    lr_G = 5e-5
    lr_D = 5e-5
    betas = (.9, .99)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr_G, betas=betas)
    C_optimizer = optim.Adam(critic.parameters(), lr=lr_D, betas=betas)
    
    trainer = WGAN(generator, critic, G_optimizer, C_optimizer, print_every=1000,
                  use_cuda=torch.cuda.is_available())
    
    trainer.train(train_loader, epochs)
      
    return generator, critic


def evaluate_upsampling(sampling_strategy, sampler):
    smote = SMOTE(sampling_strategy = sampling_target, k_neighbors=100, n_jobs=20) #random_state=123, 

    wgan_sampler = 

    result = {"F1":[],"AUC":[]}
    for sampler in [None, smote, wgan_sampler]:
        temp_result = {'F1':[],"AUC":[]}
        for _ in range(3):
            auc,f1,imb_ratio = upsampling_evaluation(X_train, X_test, y_train, y_test, 
                         #LogisticRegression(solver="lbfgs", max_iter=1e4),
                         #DecisionTreeClassifier(min_samples_leaf=50),
                         RandomForestClassifier(n_estimators=100, min_samples_leaf=100),
                         sampler)
            temp_result["F1"].append(f1)
            temp_result["AUC"].append(auc)
    
    result["F1"].append(np.mean(temp_result["F1"]))
    result["AUC"].append(np.mean(temp_result["AUC"]))