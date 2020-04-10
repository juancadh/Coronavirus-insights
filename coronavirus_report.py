import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import math
import statsmodels.api as sm
import os
plt.style.use('ggplot')

COLORS = ["#F8B195", "#F67280", "#C06C84", "#6C5B7B", "#355C7D", "#99B898", "#FECEAB", "#FF847C", "#E84A5F" ,"#2A363B"  "#A8E6CE",  "#DCEDC2",  "#FFD3B5",  "#FFAAA6",  "#FF8C94", "#A8A7A7", "#CC527A", "#E8175D", "#474747", "#363636", "#A7226E", "#EC2049", "#F26B38", "#F7DB4F", "#2F9599","#E1F5C4", "#EDE574", "#F9D423", "#FC913A", "#FF4E50","#E5FCC2", "#9DE0AD", "#45ADA8", "#547980", "#594F4F","#FE4365", "#FC9D9A", "#F9CDAD", "#C8C8A9", "#83AF9B"]

def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print("Directory " , folder_path ,  " Created ") 
    except FileExistsError:
        print("Directory " , folder_path ,  " already exists")

def get_data(savePath = ''):
    path_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

    data_confirmed = pd.read_csv(path_confirmed, index_col=1)
    data_deaths    = pd.read_csv(path_deaths, index_col=1)

    df_confirmed = transform_data(data_confirmed)
    df_deaths    = transform_data(data_deaths)

    # Transform date in confirmed
    #new_index = [(i[:len(i)-4] + i[len(i)-2:]) for i in df_confirmed.index]
    #df_confirmed.index = new_index

    # QUITAR LA ULTIMA FECHA! ( SOLO SI NO  HAY MUCHOS DATOS)
    #df_confirmed = df_confirmed.iloc[0:len(df_confirmed)-1,:]
    #df_deaths    = df_deaths.iloc[0:len(df_confirmed)-1,:]

    # Create mortality DF
    df_mortality = create_mortality_df(df_confirmed, df_deaths,  threshold_cases = 100, nRound = 3)

    # Save to CSV current data
    if len(savePath) > 0:
        df_confirmed.to_csv(f"{savePath}/confirmed.csv")
        df_deaths.to_csv(f"{savePath}/deaths.csv")

    return df_confirmed, df_deaths, df_mortality

def exponential_r2(df_confirmed, threshold = 200):

    rsq_lst = []
    for i in df_confirmed.columns:
        df_col = df_confirmed.loc[df_confirmed[i]>0,:][i]
        max_va = np.max(df_col)
        rsq = log_scale_plot(df_col, plotIt = False)
        rsq_lst.append((i,rsq, max_va))
    rsq_lsts = pd.DataFrame(rsq_lst)
    rsq_lsts.columns = ['Countries', 'RSqr', 'Max']
    rsq_lsts.set_index('Countries', inplace = True)
    rsq_lsts = rsq_lsts.loc[rsq_lsts['Max']>threshold]
    rsq_lsts = rsq_lsts.sort_values('RSqr', ascending = False)

    y_pos = np.arange(len(rsq_lsts))
    figure, ax = plt.subplots(figsize=(10,20))
    ax.barh(y_pos, rsq_lsts['RSqr'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rsq_lsts.index)
    ax.invert_yaxis()
    ax.set_xlabel('$R^2$')
    ax.set_title(f'$R^2$ - Exponential Factor')

    plt.show()

def create_mortality_df(df_confirmed, df_deaths,  threshold_cases = 100, nRound = 3):

    df_confirmed_filtered = df_confirmed.copy()
    for i in range(len(df_confirmed_filtered)):
        df_confirmed_filtered.iloc[i,:] = df_confirmed_filtered.iloc[i,:].apply(lambda x: np.nan if x < threshold_cases else x)
    df_mortality = round((df_deaths/df_confirmed_filtered)*100,nRound)
    return df_mortality

def transform_data(df):
    df_2 = df.copy()
    df_2 = df_2.iloc[:,3:]
    df_2 = df_2.transpose()
    df_2.reset_index(inplace = True)
    df_2.set_index("index", inplace = True)
    df_2.index.rename("date", inplace= True)
    df_2 = df_2.transpose().groupby("Country/Region").sum().transpose()
    return df_2

def make_ts_plot(dataframe, title = "", save_path = ""):
    figure = plt.figure(figsize=(12,6))
    plt.plot(dataframe)
    plt.xticks(rotation=90)
    plt.legend(list(dataframe.columns))
    plt.title(title)
    plt.show()
    if save_path != "":
        figure.tight_layout()
        figure.savefig(save_path, bbox_inches='tight')
    return plt

def create_factor_stats(dataframe, last_n_days = -1, plotIt = True, plt_qrt = ["max","75","50","25","min"], save_path = ""):
    factor = dataframe.shift(0)/dataframe.shift(1)
    factor.replace(np.inf, np.nan, inplace = True)

    if last_n_days == -1:
        d_factor = round(factor.describe().transpose().sort_values("50%", ascending = False),6)
        title_plt = f"Factor Stats All"
    else:
        d_factor = round(factor.tail(last_n_days).describe().transpose().sort_values("50%", ascending = False),6)
        title_plt = f"Factor Stats of last {last_n_days} days"

    if plotIt:
        figure = plt.figure(figsize=(12,6))
        ax = plt.subplot(111)

        if plt_qrt.count("max") > 0:
            plt.bar(d_factor.index, d_factor["max"], label = "max", width=0.5, color = '#2A363B')
        if plt_qrt.count("75") > 0:
            plt.bar(d_factor.index, d_factor["75%"], label = "75%", width=0.5, color = '#E84A5F')
        if plt_qrt.count("50") > 0:
            plt.bar(d_factor.index, d_factor["50%"], label = "50%", width=0.5, color = '#FF847C')
        if plt_qrt.count("25") > 0:
            plt.bar(d_factor.index, d_factor["25%"], label = "25%", width=0.5, color = '#FECEAB')
        if plt_qrt.count("min") > 0:
            plt.bar(d_factor.index, d_factor["min"], label = "min", width=0.5, color = '#99B898')

        plt.xticks(rotation=90)
        plt.legend(loc = "best")
        plt.ylim(bottom=0.95)
        plt.title(title_plt)
        plt.show()
        if save_path != "":
            figure.tight_layout()
            figure.savefig(save_path, bbox_inches='tight')

        return d_factor, plt
    else:
        return d_factor

def puntual_forecast(data, last_n_days = -1, forecast_point=1, decrease_factor_rate = 1):
    """ data: Dataframe with all the countries and data
        forecast_point: Specific day in the future to forecast based on the last value of the dataframe. 
        last_n_days: numbers of days in the past to be considered to calculate the factot.
                    -1 if want to include all the history of each country    
        decrease_factor_rate =   1 : No decrease in factor
                             (0,1) : Decrease rate speed
                                >1 : Increase in rate speed
    """

    d_factor = create_factor_stats(data, last_n_days=last_n_days, plotIt=False)
    factor_low = d_factor["25%"]*decrease_factor_rate
    factor_med = d_factor["50%"]*decrease_factor_rate
    factor_hig = d_factor["75%"]*decrease_factor_rate

    #factor_low = factor_low.apply(lambda x: 1 if x < 1 else x)
    #factor_med = factor_med.apply(lambda x: 1 if x < 1 else x)
    #factor_hig = factor_hig.apply(lambda x: 1 if x < 1 else x)

    for_low = data.iloc[-1,:]*factor_low**forecast_point
    for_med = data.iloc[-1,:]*factor_med**forecast_point
    for_hig = data.iloc[-1,:]*factor_hig**forecast_point

    return pd.DataFrame({"LCI":for_low, "MED":for_med, "HCI":for_hig})


# Number of days to forecast
def forecast_corona(data, forecast_n = 5, history_plot = 10, history = -1, highlight_country = [], plot_intervals = True, decrease_factor_speed = 1, decrease_factor_start = 1, save_path = ""):
    """ 
        decrease_factor_speed: 1 if no change. The speed of decreasing the factor over the forecast.
        decrease_factor_start  1 if no change. The starting point of forecast rate reduction. 
        Ex: if decrease_factor_speed = 0.9 and decrease_factor_start = 1
            Factor_0 = 1
            Factor_1 = 1     * 0.9  = 0.9
            Factor_2 = 0.9   * 0.9  = 0.81
            Factor_3 = 0.81  * 0.9  = 0.73
            ...

    """
    # Starting date (last day of data)
    date_str      = data.iloc[-1:].index[0]
    starting_date = datetime.strptime(date_str, "%m/%d/%y")

    # Create dataframe to store forecast
    forecast = data.tail(history_plot).transpose()
    date_columns = [datetime.strptime(i, "%m/%d/%y") for i in forecast.columns]
    forecast.columns = date_columns
    forecast = forecast.transpose().round()

    forecast_low = data.tail(1).transpose()
    forecast_med = data.tail(1).transpose()
    forecast_hig = data.tail(1).transpose()
    date_columns_2 = [datetime.strptime(i, "%m/%d/%y") for i in forecast_low.columns]

    decrease_factor_rate = decrease_factor_start
    # Start Forecast
    for f in range(1,forecast_n+1):
        mod_date = starting_date + timedelta(days=f)
        date_columns.append(mod_date)
        date_columns_2.append(mod_date)
        forecast_all = puntual_forecast(data, last_n_days=history, forecast_point=f, decrease_factor_rate = decrease_factor_rate)
        decrease_factor_rate = decrease_factor_rate * decrease_factor_speed
    
        forecast_LCI = forecast_all['LCI']
        forecast_MED = forecast_all['MED']
        forecast_HCI = forecast_all['HCI']

        forecast_low = pd.concat([forecast_low, forecast_LCI], axis = 1, sort = False)
        forecast_med = pd.concat([forecast_med, forecast_MED], axis = 1, sort = False)
        forecast_hig = pd.concat([forecast_hig, forecast_HCI], axis = 1, sort = False)

    forecast_low.columns = date_columns_2
    forecast_med.columns = date_columns_2
    forecast_hig.columns = date_columns_2

    forecast_low = forecast_low.transpose().round()
    forecast_med = forecast_med.transpose().round()
    forecast_hig = forecast_hig.transpose().round()

    def hex_to_rgb(h):
        tp = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        tp.append(0)
        tp[0] = round(tp[0]/255,1)
        tp[1] = round(tp[1]/255,1)
        tp[2] = round(tp[2]/255,1)
        return tuple(tp)

    colores = COLORS[0:len(data.columns)]
    rgb_s = [hex_to_rgb(i.lstrip('#')) for i in colores]
    
    figure = plt.figure(figsize=(15,9))
    ax1 = figure.add_subplot(111)
    plt.plot(forecast)
    if plot_intervals:
        plt.plot(forecast_low, '--', label = "Lower")
    plt.plot(forecast_med, label = "Median", linewidth=2, color="#333333")
    if plot_intervals:
        plt.plot(forecast_hig, '--', label = "Upper")
    plt.title(f"Forecast {forecast_n} days ahead")
    plt.axvline(starting_date, 0, 1, linewidth = 1, color = "#333333") 
    COUNTRIES = [] 
    for country in highlight_country:
        plt.fill_between(forecast_med.index, forecast_low[country], forecast_hig[country], color='gray', alpha=0.2)
        COUNTRIES.append(country)
    plt.legend(COUNTRIES)
    plt.show()
    if save_path != "":
        figure.tight_layout()
        figure.savefig(save_path, bbox_inches='tight')

    #fig = px.line(forecast_med, x=forecast_med.index, y="Colombia", color = )
    #fig.show()
    

    return forecast_low, forecast_med, forecast_hig


def daily_plot(data, num_days = 10, threshold_cases = 0, save_path = ""):
    df2 = dict()
    for country in data.columns:
        data_c = data[country][data[country].apply(lambda x: x > threshold_cases)].reset_index(drop = True)
        df2[country] = data_c

    df2 = pd.DataFrame(df2)
    figure = plt.figure(figsize = (12,7))
    ax1 = figure.add_subplot(111)
    ax1.plot(df2.iloc[:num_days,:])
    # Set Color
    colormap = plt.cm.get_cmap('Paired')  #nipy_spectral, Set1,Paired   gist_ncar 
    colors   = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    plt.title(f"Daily cases first {num_days} days")
    plt.legend(data.columns)
    plt.xlabel("Days")
    if save_path != "":
        figure.tight_layout()
        figure.savefig(save_path, bbox_inches='tight')
    plt.show()
    return df2, plt

def factor_evolution(data, num_last_days = 60, start_d = 5, save_path = ""):
    factor_srs = (data.shift(0)/data.shift(1))
    fat_med  = []
    fat_mean = []
    fat_25   = []
    fat_75   = []
    for i in range(start_d, num_last_days+1):
        fat_med.append(np.median(factor_srs.dropna().tail(i)))
        fat_mean.append(np.mean(factor_srs.dropna().tail(i)))
        fat_25.append(np.quantile(factor_srs.dropna().tail(i), 0.25))
        fat_75.append(np.quantile(factor_srs.dropna().tail(i), 0.75))
        
    x_ax = [i for i in range(start_d,num_last_days+1)]
    figure = plt.figure(figsize=(12,4))
    plt.plot(x_ax, fat_med, label = "Mediana", color = "#FF847C", linewidth = 2)
    #plt.plot(x_ax, fat_mean, label = "Media")
    #plt.plot(x_ax, fat_25, '--', label = "Q1", color = "#F67280")
    #plt.plot(x_ax, fat_75, '--', label = "Q3", color = "#F67280")
    plt.fill_between(x_ax, fat_25, fat_75, color='gray', alpha=0.2)
    plt.xlabel("Número acumulado de días atrás")
    plt.ylabel("Factor")
    plt.title("Evolución del factor acumulado")
    plt.legend(loc = "best")
    plt.show()
    if save_path != "":
        figure.tight_layout()
        figure.savefig(save_path, bbox_inches='tight')
    return plt

def moving_median(data, window = 7, last_n_days = 45, show_quantiles = True, spec_name = "", ylim_top = -1, plotIt = True, save_path = ""):
    factor_srs = (data.shift(0)/data.shift(1))
    factor_srs = factor_srs.tail(last_n_days)
    fdt = factor_srs.dropna().reset_index(drop=True)
    smth_med = []
    smth_25 = []
    smth_75 = []

    for i in range(len(fdt)-window):
        smth_med.append(np.median(fdt[i:i+window]))
        smth_25.append(np.quantile(fdt[i:i+window], 0.25))
        smth_75.append(np.quantile(fdt[i:i+window], 0.75))

    if plotIt:
        x_ax = factor_srs.dropna().index[0+window:len(fdt)]
        figure = plt.figure(figsize=(12,4))
        plt.plot(x_ax, smth_med, label = "Mediana", color = "#FF847C", linewidth = 2)
        if show_quantiles:
            plt.fill_between(x_ax, smth_25, smth_75, color='gray', alpha=0.2)
        plt.ylabel("Factor")
        plt.title(f"Evolución del Factor {spec_name} - Ventana {window} días")
        plt.legend(loc = "best")
        #plt.ylim(bottom=1)
        if ylim_top > 0:
            plt.ylim(top=ylim_top)
        plt.xticks(rotation=90)
        plt.show()
        if save_path != "":
            figure.tight_layout()
            figure.savefig(save_path, bbox_inches='tight')

    return smth_med

def factor_change_table(data, window=7, past_days_contrast=7):
    """
        past_days_contrast : Contrast today with <past_days_contrast> before
    """
    factors_cnt = dict()
    for i in data.columns:
        smt_f = moving_median(data.iloc[:len(data)-1,:][i], window=window, last_n_days = 40, spec_name = i, plotIt = False)
        if len(smt_f) > past_days_contrast:
            factors_cnt[i] = [smt_f[i] for i in (-past_days_contrast,-1)] # Take the last factor at current date
    factors_cnt = pd.DataFrame(factors_cnt).transpose()
    factors_cnt.columns = ['Previous', 'Today']
    factors_cnt['Change'] = round(((factors_cnt['Today']-factors_cnt['Previous'])/factors_cnt['Previous'])*100,1)
    factors_cnt = factors_cnt.sort_values('Change')
    print(f"Previous data correspond to: {past_days_contrast} days ago")
    return factors_cnt

def factor_plot(data, window = 7, var_name = 'Today', past_days_contrast=7, save_path = ""):
    """
        var_name: Could be 'Today' or 'Change'
        past_days_contrast : Contrast today with <past_days_contrast> before
    """
    factor_change = factor_change_table(data, window=window, past_days_contrast=past_days_contrast)
    factor_change = factor_change.sort_values(var_name)
    factor_change['positive'] = factor_change[var_name] > 0
    y_pos = np.arange(len(factor_change))
    figure, ax = plt.subplots(figsize=(10,20))
    ax.barh(y_pos, factor_change[var_name], color=factor_change.positive.map({True: '#E84A5F', False: '#99B898'}))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factor_change.index)
    ax.invert_yaxis()
    ax.set_xlabel('Change in Factor')
    ax.set_title(f'Percental Change in Factor (Today vs {past_days_contrast} days ago)')
    if var_name == 'Today':
        ax.set_xlabel('Factor ($I_{n+1}$/$I_n$)')
        ax.set_title(f'Factor Today')
        ax.set_xlim([0.9,2])
    plt.show()
    if save_path != "":
        figure.tight_layout()
        figure.savefig(save_path, bbox_inches='tight')

    return factor_change

def dynamic_evolution(data, var_name = "Cases", number_of_countries = 10):

    fig = go.Figure()
    # Add traces, one for each slider step
    for step in np.arange(len(data), 0, -1):
        last_data_conf   = data.iloc[-step,:].sort_values(ascending=False).head(number_of_countries)
        last_data_conf = pd.DataFrame(last_data_conf)
        last_data_conf.reset_index(inplace = True)
        date_i = last_data_conf.columns[1]
        last_data_conf.columns = ['Country', 'Cases']

        fig.add_trace(
            go.Bar(
                visible = False,
                name = date_i,
                y = last_data_conf['Country'],
                x = last_data_conf['Cases'],
                text=last_data_conf['Cases'],
                orientation='h',
                opacity=0.8
            )
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')

    fig.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=f'Date: {data.index[i]} - Days ago: {len(fig.data)-i}'
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders = sliders,
        yaxis   = dict(autorange="reversed"),
        width   = 570,
        height  = 650,
        margin  = dict(l=150, r=30, t=50, b=20),
        title   = f"Evolution of {var_name}"
    )

    fig.show()

def log_scale_plot(data, series_title = "Series", save_path = "", plotIt = True):
    """ data : Just one series """
    df_log = np.log10(data)
    Y = df_log
    X = [i for i in range(len(Y))]
    results = sm.OLS(Y,sm.add_constant(X)).fit()
    RSq_adj = np.round(results.rsquared_adj,3)

    if plotIt:
        X_plot = np.array(X)
        figure = plt.figure(figsize=(12,6))
        plt.plot(df_log.index, df_log)
        plt.plot(df_log.index, X_plot * results.params[1] + results.params[0], '--', color = "#594F4F")
        plt.xticks(rotation=90)
        plt.title(f"{series_title} - Log - $R^2 = {RSq_adj}$")
        plt.show()

        if save_path != "":
            figure.tight_layout()
            figure.savefig(save_path, bbox_inches='tight')
        return plt, RSq_adj
    else:
        return RSq_adj