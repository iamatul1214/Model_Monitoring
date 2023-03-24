def psi(score_initial:list, score_new:list, num_bins = 10, mode = 'fixed'):
    """
    This method calculates the psi score for the training distribution and prod distribution.
    ----Inputs---
    score_inital: The distirbution which was used at time of training the model.
    score_new: The distribution observed from the prod enviornment.
    num_bins: The total number of buckets required for analysis. Default = 10.
    mode: The mode of buckets required for analysis. It can only be 'fixed' or 'quantile'. Default = 'Fixed'
    
    ---Returns---
    This method will create buckets, and calculate the psi and return the dataframe with psi of each bucket and overall psi.
    """
    epsilon = 1e-4
    
    # Sort the data
    score_initial.sort()
    score_new.sort()
    
    # Prepare the bins
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == 'fixed':
        bins = [min_val + (max_val - min_val)*(i)/num_bins for i in range(num_bins+1)]
    elif mode == 'quantile':
        bins = pd.qcut(score_initial, q = num_bins, retbins = True)[1] # Create the quantiles based on the initial population
    else:
        raise ValueError(f"Mode \'{mode}\' not recognized. Your options are \'fixed\' and \'quantile\'")
    bins[0] = min_val - epsilon # Correct the lower boundary
    bins[-1] = max_val + epsilon # Correct the higher boundary
        
        
    # Bucketize the initial population and count the sample inside each bucket
    bins_initial = pd.cut(score_initial, bins = bins, labels = range(1,num_bins+1))
    df_initial = pd.DataFrame({'train': score_initial, 'bin': bins_initial})
    grp_initial = df_initial.groupby('bin').count()
    grp_initial['percent_train'] = grp_initial['train'] / sum(grp_initial['train'])
    
    # Bucketize the new population and count the sample inside each bucket
    bins_new = pd.cut(score_new, bins = bins, labels = range(1,num_bins+1))
    df_new = pd.DataFrame({'prod': score_new, 'bin': bins_new})
    grp_new = df_new.groupby('bin').count()
    grp_new['percent_prod'] = grp_new['prod'] / sum(grp_new['prod'])
    
    # Compare the bins to calculate PSI
    psi_df = grp_initial.join(grp_new, on = "bin", how = "inner")
    
    # Add a small value for when the percent is zero
    psi_df['percent_train'] = psi_df['percent_train'].apply(lambda x: epsilon if x == 0 else x)
    psi_df['percent_prod'] = psi_df['percent_prod'].apply(lambda x: epsilon if x == 0 else x)
    
    # Calculate the psi
    psi_df['psi'] = (psi_df['percent_train'] - psi_df['percent_prod']) * np.log(psi_df['percent_train'] / psi_df['percent_prod'])
    
    # Return the psi values
    # print(psi_df)
    return psi_df,psi_df['psi'].values


def plot_train_vs_prod(df:pd.DataFrame,train_distribution_col:str,prod_distribution_col:str,title:str)->None:
    """
    This method is used to plot the training and prod distribution of the features/scores together.
    ---Inputs---
    df: The main dataframe which contains the bucketized distribution of training and prod feature.
    train_distribution_col: The name of the column of dataframe which represents training bucket distribution.
    prod_distribution_col: The name of the column of dataframe which represents prod bucket distribution.
    """
    
    df[[train_distribution_col,prod_distribution_col]].plot(kind = 'bar')
    plot_title = f"{title} train vs prod"
    plt.title(plot_title)
    plt.ylabel("score frequency")
    plt.xlabel("score buckets")
    plt.rcParams["figure.figsize"] = (15,15)


def calculate_psi_overall(training_probs,prod_probs):
    return (sum((training_probs[i]-prod_probs[i])*log(training_probs[i]/prod_probs[i]) for i in range(len(training_probs))))/len(training_probs)
    

def calculate_kl_divergence_overall(trained_scores,prod_scores):
    return (sum(trained_scores[i] * log(trained_scores[i]/prod_scores[i]) for i in range(len(trained_scores))))/len(trained_scores)


def compare_central_tendency(training_values:list, new_values:list):
    from collections import Counter
    ## calculating mean 
    training_values_mean = float(sum(training_values)/len(training_values))
    new_values_mean = float(sum(new_values)/len(new_values))
    
    ## calculating median
    training_values.sort()
    new_values.sort()
    n = len(training_values)
    if n % 2 == 0:
        training_values_median = float((training_values[n//2] + training_values[n//2 - 1])/2)
    training_values_median = float(training_values[n//2])
    n = len(new_values)
    if n% 2==0:
        new_values_median = float((new_values[n//2]+new_values[n//2 - 1])/2)
    new_values_median = float(new_values[n//2])
    
    ## calculating the mode
    freq = Counter(training_values)
    training_values_mode = [k for k,v in freq.items() if v == freq.most_common(1)[0][1]]
    if len(training_values_mode)==1:
        training_values_mode = training_values_mode[0]
    freq = Counter(new_values)
    new_values_mode = [k for k,v in freq.items() if v == freq.most_common(1)[0][1]]
    if len(new_values_mode)==1:
        new_values_mode = new_values_mode[0]
    
    
     ## creating dataframe for storing the central trendecies
    index = ['mean','median','mode']
    train_tendecy = [training_values_mean,training_values_median,training_values_mode]
    new_tendency = [new_values_mean,new_values_median,new_values_mode]
    
    df = pd.DataFrame({'training (initial)':train_tendecy,'Prod (newly observed)':new_tendency}, index=index)
    display(df)
    return df