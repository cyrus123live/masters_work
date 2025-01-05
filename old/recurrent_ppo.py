elif model_type == "Recurrent_PPO":
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh, 
            net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
            lstm_hidden_size=512, 
            n_lstm_layers=1, 
            shared_lstm=True, 
            enable_critic_lstm=False
        )
        model = RecurrentPPO("MlpLstmPolicy", 
            train_env, 
            verbose=0, 
            seed=seed, 
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            n_steps=128,
            vf_coef=0.5,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            learning_rate=3e-4
        )
        model.policy.optimizer = optim.Adam(
            model.policy.parameters(),
            lr=3e-4,              # Learning Rate
            betas=(0.9, 0.999),   # β₁ and β₂
            eps=1e-8              # ε
