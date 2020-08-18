import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def plot_data(x, y):
    plt.hist2d(x, y, bins=50, range=np.array([(-1, 1), (-1, 1)]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def eval_plot(plot=False):
    data_path = './data_artificial/'
    df_naive = pd.read_csv(data_path+'artificial_raw.csv')
    df_naive.columns = df_naive.columns.astype(int)
    tfs_A_samples = pd.read_csv(data_path+'artificial_tfs_a_2.csv')
    tfs_A_samples.columns = tfs_A_samples.columns.astype(int)
    tfs_B_samples = pd.read_csv(data_path+'artificial_tfs_b.csv')
    tfs_B_samples.columns = tfs_B_samples.columns.astype(int)
    b1_samples = pd.read_csv(data_path+'artificial_b1.csv')
    b1_samples.columns = b1_samples.columns.astype(int)
    b4_samples = pd.read_csv(data_path+'artificial_b4.csv')
    b4_samples.columns = b4_samples.columns.astype(int)
    tfs_C_samples = pd.read_csv(data_path+'artificial_tfs_prior.csv')
    tfs_C_samples.columns = tfs_C_samples.columns.astype(int)

    ########################################################################
    # plotting gen data
    ########################################################################
    print('#'*30 + 'corr_xy' + '#'*30)
    corr_raw = np.corrcoef(df_naive[0], df_naive[1])[0, 1]
    corr_tfs_A = np.corrcoef(tfs_A_samples[0], tfs_A_samples[1])[0, 1]
    corr_tfs_B = np.corrcoef(tfs_B_samples[0], tfs_B_samples[1])[0, 1]
    corr_b1 = np.corrcoef(b1_samples[0], b1_samples[1])[0, 1]
    corr_b4 = np.corrcoef(b4_samples[0], b4_samples[1])[0, 1]
    print('xy_corr_raw', corr_raw)
    print('xy_corr_tfs_A', corr_tfs_A)
    print('xy_corr_tfs_B', corr_tfs_B)
    print('xy_corr_b1', corr_b1)
    print('xy_corr_b4', corr_b4)

    print('#'*30 + 'autocorr_x' + '#'*30)
    print('x_autocorr_raw', estimated_autocorrelation(df_naive[0].to_numpy())[:5])
    print('x_autocorr_tfs_A', estimated_autocorrelation(tfs_A_samples[0].to_numpy())[:5])
    print('x_autocorr_tfs_B', estimated_autocorrelation(tfs_B_samples[0].to_numpy())[:5])
    print('x_autocorr_b1', estimated_autocorrelation(b1_samples[0].to_numpy())[:5])
    print('x_autocorr_b4', estimated_autocorrelation(b4_samples[0].to_numpy())[:5])

    if plot:
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 3, 1)
        plot_data(df_naive[0].to_numpy(), df_naive[1].to_numpy())
        plt.title("Observed Data")
        plt.subplot(3, 3, 2)
        plot_data(tfs_A_samples[0].to_numpy(), tfs_A_samples[1].to_numpy())
        plt.title("TFS_A Sampled Data")
        plt.subplot(3, 3, 3)
        plot_data(tfs_B_samples[0].to_numpy(), tfs_B_samples[1].to_numpy())
        plt.title("TFS_B Sampled Data")
        plt.subplot(3, 3, 4)
        plot_data(b1_samples[0].to_numpy(), b1_samples[1].to_numpy())
        plt.title("B1 Sampled Data")
        plt.subplot(3, 3, 5)
        plot_data(b4_samples[0].to_numpy(), b4_samples[1].to_numpy())
        plt.title("B4 Sampled Data")
        plt.subplot(3, 3, 6)
        plot_data(tfs_C_samples[0].to_numpy(), tfs_C_samples[1].to_numpy())
        plt.title("TFS_C Sampled Data") 
        
        plt.savefig('artificial_data_cmin.png')
        plt.show()

def eval_task_col():
    data_path = './data_artificial/'
    print('='*30 + 'eval_task_row' + '='*30)
    def mse_xy(data_name):
        df = pd.read_csv(data_path+f'artificial_{data_name}.csv')
        df.columns = df.columns.astype(int) 
        df_raw = pd.read_csv(data_path+'artificial_raw.csv')
        df_raw.columns = df_raw.columns.astype(int) 

        x = np.reshape(df[0].tolist(), (-1, 1))
        y = np.reshape(df[1].tolist(), (-1, 1))
        x_true = np.reshape(df_raw[0].tolist(), (-1, 1))
        y_true = np.reshape(df_raw[1].tolist(), (-1, 1)) 
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_true)
        print(f'{data_name}_train_mse', mean_squared_error(y_true, y_pred))
        # print('real_train_rmse', mean_squared_error(real_y, pred_y, squared=False))
    mse_xy('raw')
    mse_xy('tfs_prior_2')
    mse_xy('tfs_a_2')
    mse_xy('tfs_b')
    mse_xy('b1')
    mse_xy('b4')

def eval_task_row():
    data_path = './data_artificial/'
    print('='*30 + 'eval_task_col' + '='*30)
    def mse_x(data_name):
        df = pd.read_csv(data_path+f'artificial_{data_name}.csv')
        df.columns = df.columns.astype(int)
        df_len = len(df.columns)
        df[df_len] = df[0].shift(-1)
        df = df.dropna()

        df_raw = pd.read_csv(data_path+'artificial_raw.csv')
        df_raw.columns = df_raw.columns.astype(int) 
        df_raw[df_len] = df_raw[0].shift(-1)
        df_raw = df_raw.dropna()

        x = df[[0, 1]].to_numpy() #np.reshape(df['raw_x'].tolist(), (-1, 1))
        y = df[[2]].to_numpy() 
        x_true = df_raw[[0, 1]].to_numpy() #np.reshape(df['raw_x'].tolist(), (-1, 1))
        y_true = df_raw[[2]].to_numpy() 
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_true)
        print(f'{data_name}_train_mse', mean_squared_error(y_true, y_pred))
        # print('real_train_rmse', mean_squared_error(real_y, pred_y, squared=False))
    mse_x('raw')
    mse_x('tfs_a')
    mse_x('tfs_b')
    mse_x('b1')
    mse_x('b4')

if __name__ == "__main__":
    eval_plot(plot=True)
    # eval_task_row()
    eval_task_col()