from wandb.keras import WandbCallback
import tensorflow as tf

class cd_wandb_custom(WandbCallback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs):

        np_predictions = self.model.predict(self.ds_test)
        self.np_ds_test["predictions"] = np_predictions
        sample_data = self.__get_index_data(
            np_ds_test=self.np_ds_test, test_idx=self.test_idx
        )

        _, sample_fig = self.__plot_scene(features=sample_data)

        if (epoch-1) % 2 == 0:
            wandb.log({f"sample_img_{epoch-1}": sample_fig})
            sample_fig.close()

        return super().on_epoch_begin(epoch, logs=logs)
        
    def get_filter_plot(self, layer):
        for i, channel in enumerate(CHANNEL):
            for j, filter in enumerate(FILTER):
                
                f = filters[:,:,:, filter]
                f = np.array(f[:,:, channel])  
                print(f)
                theta = getSobelAngle(f)
                theta = theta[theta.shape[0]//2, theta.shape[1]//2]
                thetas.append(theta)
                mag = np.linalg.norm(f) 

                mags.append(mag)

                plt.hist(thetas, bins=16)
                plt.xticks(np.arange(0, 2*np.pi, step=1), size='small', rotation=0)    
                plt.title("Layer {}, Filter = {}, Channel orientation Distribution".format("N", FILTER[0]))
                plt.xlabel('θ (Deg)')
                plt.ylabel('Count')
                plt.show()
                print(len(thetas))
                t_rad = thetas
                n = len(thetas)
                r = np.sqrt(np.sum(np.cos(t_rad))**2 + np.sum(np.sin(t_rad))**2 )
                print(1 - r/n)

                plt.hist(mags, bins=32)
                plt.xticks(np.arange(np.min(mags), np.max(mags), step=45), size='small', rotation=0)    
                plt.xlabel('magnitude ')
                plt.ylabel('Count')
                plt.show()
                len(mags)
        
model = tf.keras.models.Sequential([
tf.keras.Input(name='input_layer', shape=(10,)),
tf.keras.layers.Dense(50,activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')])
   
opt = 'adam'
model.compile(opt)
# Learning rate scheduler
cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=0.0001)

history = model.fit(ds_train,epochs=params.get("epochs"),validation_data=ds_valid,
    callbacks=[cb_reduce_lr, cd_wandb_custom(ds_test=ds_test, np_test_dataset=np_ds_test,test_index=15),],)
      