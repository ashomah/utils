def pipeline(model, df, target, scoring='recall', split_ratio=0.2, n_splits=10, seed=888, dump_path=os.path.join('models'), model_name='testrun', silent=False, grid=None):
    
    start_time = datetime.now()
    
    # Train/Test Split
    X, y, X_train, X_test, y_train, y_test = split_dataset(df, target, split_ratio, seed)
    
    if grid == None:
        # Training
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        # Cross-Validation
        cv_results = cv_kfold(model, X_train, y_train, scoring, n_splits, seed)

        # Get Coefs
        try:
            coefs = pd.concat([pd.DataFrame(X_train.columns, columns=['Feature']), pd.DataFrame(np.transpose(model.coef_), columns=['Coef'])], axis = 1)
            coefs['Abs'] = abs(coefs.Coef.loc[coefs.Feature != 'intercept'])
            coefs['Rank'] = abs(coefs.Coef.loc[coefs.Feature != 'intercept']).rank(method='first',ascending=False)
            coefs['Rank'] = coefs['Rank'].astype(int)
            coefs.loc[-1] = ['intercept', model.intercept_[0], abs(model.intercept_[0]), 0]
            coefs.index = coefs.index + 1
            coefs = coefs.sort_index()
        except:
            try:
                coefs = pd.concat([pd.DataFrame(X_train.columns, columns=['Feature']), pd.DataFrame(np.transpose(model.feature_importances_), columns=['Coef'])], axis = 1)
                coefs['Abs'] = abs(coefs.Coef.loc[coefs.Feature != 'intercept'])
                coefs['Rank'] = abs(coefs.Coef.loc[coefs.Feature != 'intercept']).rank(method='first',ascending=False)
                coefs['Rank'] = coefs['Rank'].astype(int)
                coefs.index = coefs.index + 1
                coefs = coefs.sort_index()
            except:
                coefs = model.feature_importances_

        # Save Model
        end_time = datetime.now()
        model_file = model_name + '_' + end_time.strftime('%y%m%d_%H%M%S') + '_model.joblib'
        dump(model, os.path.join(dump_path, model_file))

        training_time = end_time - start_time
        training_time_hours = int(training_time.total_seconds() // 3600)
        training_time_minutes = int((training_time.total_seconds() - training_time_hours * 3600) // 60)
        training_time_seconds = int(training_time.total_seconds() - training_time_hours * 3600 - training_time_minutes * 60)

        dict_results = {'score': score,
                        'accuracy':accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'cv_mean': cv_results.mean(),
                        'cv_results': cv_results,
                        'coefs': coefs,
                        'count_coefs': len(coefs),
                        'training_time': training_time}
        results_file = model_name + '_' + end_time.strftime('%y%m%d_%H%M%S') + '_results.p'
        with open(os.path.join(dump_path, results_file), 'wb') as file:
            pickle.dump(dict_results, file, protocol=pickle.HIGHEST_PROTOCOL)

        if not silent:
            print('{}{:<17} {:>.3f}'.format(scoring.capitalize(), ' on Test:', dict_results[scoring]))
            print('{:02d}{:14}{}: {:>.3f}'.format(n_splits, '-Fold CV Mean ', scoring.capitalize(), cv_results.mean()))
            str_cv_results = str({i+1: np.round(cv_results[i], 3) for i in range(len(cv_results))})[1:-1]
            print('{:<10} [{}]\n'.format('Iterations:', str_cv_results))
            print('Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}\n'.format(accuracy, precision, recall, f1))
            print('Training Time: {:02d}:{:02d}:{:02d}'.format(training_time_hours, training_time_minutes, training_time_seconds))
            print()
            print(classification_report(y_test, y_pred))

            fig, ax =plt.subplots(1,2, sharey=False, figsize=(20,4)) 
            plot_confusion_matrix(model, X_test, y_test, normalize=None, cmap=plt.cm.Blues, values_format=',', ax=ax[0])
            plot_roc_curve(model, X_test, y_test, ax=ax[1])
        return dict_results
    else:
        model.fit(X_train,y_train.values.ravel())
        print(model.best_score_)
        print(model.best_params_)
