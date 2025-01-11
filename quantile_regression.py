import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class QuantileRegressionModel:
    def __init__(self, data, quantiles, factors, formula_template):
        self.data = data
        self.quantiles = quantiles
        self.factors = factors
        self.formula_template = formula_template
        self.models = {}
        self.build_models()

    def fit_model(self, q, formula):
        mod = smf.quantreg(formula, self.data)
        return mod.fit(q=q)

    def fit_ols_model(self, formula):
        ols = smf.ols(formula, self.data).fit()
        return ols

    def build_models(self):
        for factor in self.factors:
            formula = self.formula_template.format(factor=factor)
            self.models[factor] = {str(q): self.fit_model(q, formula) for q in self.quantiles}
            self.models[factor]['ols'] = self.fit_ols_model(formula)

    def plot_results(self, mode='line', indp_var= 'Quantile_Num'):
        num_factors = len(self.factors)
        num_cols = 4
        num_rows = num_factors // num_cols + (num_factors % num_cols > 0)
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink', 'teal']
        income_range = self.data[indp_var]

        plt.figure(figsize=(20, 5 * num_rows))

        for i, factor in enumerate(self.factors, 1):
            plt.subplot(num_rows, num_cols, i)
            if mode == 'scatter':
                sns.scatterplot(data=self.data, x=indp_var, y=factor, color='gray', alpha=0.5)

                for i, q in enumerate(self.quantiles):
                    plt.plot(income_range, self.models[f'{factor}'][str(q)].params['Intercept'] + self.models[f'{factor}'][str(q)].params[indp_var] * income_range,
                            label=f'{q}', color=colors[i % len(colors)])
            elif mode == 'line':
                x = list(self.models[factor].keys())[:-1]
                quan_params = [self.models[factor][str(q)].params[indp_var] for q in self.quantiles]
                quan_lb = [self.models[factor][str(q)].conf_int().loc[indp_var][0] for q in self.quantiles]
                quan_ub = [self.models[factor][str(q)].conf_int().loc[indp_var][1] for q in self.quantiles]
                ols_b = [self.models[factor]["ols"].params[indp_var]]
                ols_lb = [self.models[factor]["ols"].conf_int().loc[indp_var][0]]
                ols_ub = [self.models[factor]["ols"].conf_int().loc[indp_var][1]]

                plt.plot(x, quan_params, color="black", label="Quantile Reg.")
                plt.fill_between(x, quan_lb, quan_ub, color='grey', alpha=0.5)
                plt.plot(x, ols_b * len(x), color="red", label="OLS")
                plt.plot(x, ols_lb * len(x), linestyle="dotted", color="red")
                plt.plot(x, ols_ub * len(x), linestyle="dotted", color="red")
            else:
                raise ValueError("Invalid mode. Use 'scatter' or 'line'.")

            plt.title(f'{str(factor).upper()}')
            plt.xlabel('Group')
            plt.ylabel('CAGR (%)')
            plt.legend([],[], frameon=False)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def get_coefficients_and_pvalues(self, indp_var= 'Quantile_Num', ols= True):
        reg_result = {}
        reg_pvalue = {}
        quantiles = self.quantiles
        if ols:
            quantiles.append('ols')


        for f in self.factors:
            coif_temp = []
            pval_temp = []
            for q in self.models[f].keys():
                if ols:
                    coif_temp.append(self.models[f][q].params[indp_var])
                    pval_temp.append(self.models[f][q].pvalues[indp_var])
                else:
                    if q != 'ols':  # 跳過 OLS 結果
                        coif_temp.append(self.models[f][q].params[indp_var])
                        pval_temp.append(self.models[f][q].pvalues[indp_var])
            reg_result[f] = coif_temp
            reg_pvalue[f] = pval_temp

        reg_result_df = pd.DataFrame.from_dict(reg_result).T
        reg_result_df.columns = quantiles

        reg_pvalue_df = pd.DataFrame.from_dict(reg_pvalue).T
        reg_pvalue_df.columns = quantiles
        return reg_result_df, reg_pvalue_df
