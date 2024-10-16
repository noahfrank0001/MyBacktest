import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf

class FindData:
    def __init__(self, tick):
        self.tick = tick

    
    def get_data(self, period="5y"):
        self.data = yf.Ticker(self.tick).history(period=period)

        return self.data

class DataClean:
    def __init__(self, data):
        self.data = data


    def prepare_data(self):
        self.data = self.data.drop(columns=["Dividends","Stock Splits"])
        return self.data
    

    def create_target(self, scale, col="Close", days_ahead=1, range=5):
        inp_df = self.data

        if scale == "binary":
            inp_df["Target"] = np.where(inp_df[col] < inp_df[col].shift(-days_ahead), 1, 0)

        if scale == "cont":
            inp_df["Target"] = np.where((inp_df[col].shift(-days_ahead) / inp_df[col] - 1) * 100 > range, range,
                                    np.where((inp_df[col].shift(-days_ahead) / inp_df[col] - 1) * 100 < -range, -range, 
                                    np.where(abs(round((inp_df[col].shift(-days_ahead) / inp_df[col] - 1) * 100, 0)) == 0, 0,
                                    round((inp_df[col].shift(-days_ahead) / inp_df[col] - 1) * 100, 0)
            )))

        inp_df["Target"] = inp_df["Target"].fillna(0).astype(int)

        drop_cols = ["Target", "Open", "High", "Low", "Close"]

        return inp_df #, drop_cols
    

    def calculate_avg(self, periods=[5, 10, 30], cols="all", drop_na=True, drop_org_cols=True):    

        # inp_df = data.copy
        # cols = ["Open", "High", "Low", "Close", "Volume"]
        # periods = [5, 10, 30]

        inp_df = self.data

        use_cols = ""
        if cols == "all":
            use_cols = ["Open", "High", "Low", "Close"]

        for col in use_cols:
            for period in periods:
                inp_df[f"{col}_MA_{period}"] = inp_df[col] / inp_df[col].rolling(period).mean()
                
        if drop_na:
            inp_df = inp_df.dropna()

        # if drop_org_cols:
        #     inp_df = inp_df.drop(columns=["Open", "High", "Low", "Close"])

        return inp_df
    
    
    def full_pipeline(self):
        data = self.data

        data = self.prepare_data()
        data = self.create_target("cont")
        data = self.calculate_avg()

        return data



class Backtest:
    def __init__(self, data):
        self.data = data

    def get_train_data(self, test_size):

        data = self.data

        X = data.iloc[0:len(data) - test_size].drop(columns=["Target", "Open", "High", "Low", "Close"])
        y = data.iloc[0:len(data) - test_size]["Target"]

        return X, y


    def get_money(self, sell_shares, df):
        """
        NEED TO EVENTUALLY TAKE INTO ACCOUNT SHARES THAT WERE ADDED WHEN OBJECT WAS MADE

        Used to ensure that we have the right amount in "Invested". This is important because it 
        determines our return. 
        """
        back_index = ''
        offset = 0

        for index in df.index[::-1]:
            # historical_buy_df[index:]["Shares_Bought"]
            if df.loc[index:]["Share Effect"].sum() == sell_shares:
                # print(historical_buy_df[index:]["Shares_Bought"])
                back_index = index
                break
            elif df.loc[index:]["Share Effect"].sum() > sell_shares:
                back_index = index
                offset = df.loc[index:]["Share Effect"].sum() - sell_shares
                break

            

        # share_effect = 0
        money_effect = 0
        counter = 0

        for index in df.loc[back_index:].index:
            counter += 1
            if counter == 1:
                money_effect += (df.loc[index]["Share Effect"] - offset) * df.loc[index]["Current Close"]

            else: 
                money_effect += df.loc[index]["Share Effect"] * df.loc[index]["Current Close"]

        return money_effect


    def backtest(self, models={}, cash=10000, test_size=0, weekly_injection=0, share_multiplier={"buy": 5, "sell": 5}, plot_market=False):
        data = self.data

        if test_size == 0:
            test_size = math.floor(len(self.data) // 2)

        if models == {}:
            ### THROW ERROR - NO MODEL INPUT

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier

            X = data.iloc[0:len(data) - test_size].drop(columns=["Target", "Open", "High", "Low", "Close"])
            y = data.iloc[0:len(data) - test_size]["Target"]

            knn_clf = KNeighborsClassifier().fit(X, y)
            tree_clf = DecisionTreeClassifier(random_state=42).fit(X, y)
            rf_clf = RandomForestClassifier(random_state=42).fit(X, y)

            models = {
            "KNN": knn_clf,
            "Tree": tree_clf,
            "Random Forrest": rf_clf
            # "Log-Reg": lr_clf
            }
        
        # if models != {}: ### USER NEEDS TO GRIDSEARCH BEFORE INPUTTING
            

        original_cash = cash
        self.original_data = data.iloc[test_size:]
        results_df = pd.DataFrame()
        self.log = []

        ### NEED TO FOR LOOP THIS FOR EACH TRAINED MODEL ###
        for mod_name, model in models.items():
            
            cash = original_cash
            cash_injections = 0
            current_shares = 0
            total_invested = 0
            current_equity = 0
            investment_days = 0
            potential_shares = 0
            data = self.original_data
            log_df = pd.DataFrame()

            # log_df = pd.DataFrame()

            for index, row in data.iterrows():
                
                investment_days += 1

                current_row = data.loc[index:index]
                current_close = current_row[["Close"]].to_numpy()[0][0]

                if weekly_injection == -1 and investment_days % 5 == 0:
                    cash += current_close
                    cash_injections += current_close

                elif weekly_injection > 0 and investment_days % 5 == 0:
                    cash += weekly_injection
                    cash_injections += weekly_injection

                X_inner = current_row.drop(columns=["Target", "Open", "High", "Low", "Close"])
                y_inner = current_row["Target"]
                pred = model.predict(X_inner)

                # share_multiplier = 5

                if pred[0] >= 1:

                    # Invest - Adequate Money
                    if cash - (abs(pred[0]) * current_close * share_multiplier["buy"]) >= 0:
                        current_shares += (abs(pred[0]) * share_multiplier["buy"])
                        total_invested += ((abs(pred[0]) * share_multiplier["buy"]) * current_close)
                        current_equity = current_shares * current_close
                        cash = cash - ((abs(pred[0]) * share_multiplier["buy"]) * current_close)

                        # Logging our trades
                        log_dict = {
                            "Date": [index],
                            "Investment Day": [investment_days],
                            "Action": ["Invest - Adequate Money"],
                            "Model Projection": [pred[0]],
                            "Share Effect": [pred[0] * share_multiplier["buy"]],
                            "Cash Effect": [-(pred[0] * share_multiplier["buy"] * current_close)],
                            "Current Close": [current_close],
                            "Current_Shares": [current_shares],
                            "Current_Cash": [cash],
                            "Cash_Injections": [cash_injections],
                            "Total_Invested": [total_invested],
                            "Current_Equity": [current_equity]
                        }

                        log_df = pd.concat([log_df, pd.DataFrame.from_dict(log_dict, orient="columns")])

                    # Invest - Inadequate Money
                    else: 
                        current_shares += math.floor(cash // current_close)
                        total_invested += (math.floor(cash // current_close) * current_close)
                        current_equity = current_shares * current_close
                        cash -= math.floor(cash // current_close) * current_close

                        # Logging our trades
                        log_dict = {
                            "Date": [index],
                            "Investment Day": [investment_days],
                            "Action": ["Invest - Inadequate Money"],
                            "Model Projection": [pred[0]],
                            "Share Effect": [math.floor(cash // current_close)],
                            "Cash Effect": [-(math.floor(cash // current_close) * current_close)],
                            "Current Close": [current_close],
                            "Current_Shares": [current_shares],
                            "Current_Cash": [cash],
                            "Cash_Injections": [cash_injections],
                            "Total_Invested": [total_invested],
                            "Current_Equity": [current_equity]
                        }

                        log_df = pd.concat([log_df, pd.DataFrame.from_dict(log_dict, orient="columns")])

                if pred[0] <= -1:

                    # Sell - Adequate Shares
                    if current_shares >= (abs(pred[0]) * share_multiplier["sell"]):

                        buy_log = log_df[log_df["Share Effect"] > 0][["Share Effect", "Current Close"]]

                        current_shares -= (abs(pred[0]) * share_multiplier["sell"])
                        total_invested -= (self.get_money(abs(pred[0]) * share_multiplier["sell"], buy_log.reset_index())) ### NEED TO ADD HERE
                        current_equity = current_shares * current_close
                        cash += ((abs(pred[0]) * share_multiplier["sell"]) * current_close)


                        # Logging our trades
                        log_dict = {
                            "Date": [index],
                            "Investment Day": [investment_days],
                            "Action": ["Sell - Adequate Shares"],
                            "Model Projection": [pred[0]],
                            "Share Effect": [-(abs(pred[0]) * share_multiplier["sell"])],
                            "Cash Effect": [(abs(pred[0]) * share_multiplier["sell"]) * current_close],
                            "Current Close": [current_close],
                            "Current_Shares": [current_shares],
                            "Current_Cash": [cash],
                            "Cash_Injections": [cash_injections],
                            "Total_Invested": [total_invested],
                            "Current_Equity": [current_equity]
                        }

                        log_df = pd.concat([log_df, pd.DataFrame.from_dict(log_dict, orient="columns")])

                    # Sell - Inadequate Shares
                    elif 0 < current_shares < (abs(pred[0]) * share_multiplier["sell"]):
                        cur_share = current_shares
                        cash_gain = current_shares * current_close

                        total_invested = 0
                        current_equity = 0
                        cash += (current_shares * current_close)
                        current_shares = 0

                        # Logging our trades
                        log_dict = {
                            "Date": [index],
                            "Investment Day": [investment_days],
                            "Action": ["Sell - Inadequate Shares"],
                            "Model Projection": [pred[0]],
                            "Share Effect": [-cur_share],
                            "Cash Effect": [cash_gain],
                            "Current Close": [current_close],
                            "Current_Shares": [current_shares],
                            "Current_Cash": [cash],
                            "Cash_Injections": [cash_injections],
                            "Total_Invested": [total_invested],
                            "Current_Equity": [current_equity]
                        }

                        log_df = pd.concat([log_df, pd.DataFrame.from_dict(log_dict, orient="columns")])

                if pred[0] == 0:
                    log_dict = {
                            "Date": [index],
                            "Investment Day": [investment_days],
                            "Action": ["No Action"],
                            "Model Projection": [pred[0]],
                            "Share Effect": [0],
                            "Cash Effect": [0],
                            "Current Close": [current_close],
                            "Current_Shares": [current_shares],
                            "Current_Cash": [cash],
                            "Cash_Injections": [cash_injections],
                            "Total_Invested": [total_invested],
                            "Current_Equity": [current_equity]
                        }

                    log_df = pd.concat([log_df, pd.DataFrame.from_dict(log_dict, orient="columns")])

                potential_shares = math.floor(cash / current_close)

            investment_return = 0
            if total_invested > 0:
                investment_return = current_equity / total_invested

            market_increase = data.iloc[-1]["Close"] / data.iloc[0]["Close"]

            results_dict = {
                "Current_Shares": [current_shares],
                "Current_Cash": [cash],
                "Cash_Injections": [cash_injections],
                "Potential_Shares": [potential_shares],
                "Investment_Period": [investment_days],
                "Total_Invested": [total_invested],
                "Current_Equity": [current_equity],
                "Total_Assets": [cash + current_equity],
                "Return": [investment_return],
                "Market_Increase": [market_increase],
                "Return_vs_Market": [investment_return - market_increase],
                "Asset_Return": [(cash + current_equity) / (original_cash + cash_injections)]
            }

            log_df.index = log_df["Date"]
            log_df = log_df.drop(columns=["Date"])

            self.log.append(log_df)

            current_reults_df = pd.DataFrame.from_dict(results_dict, orient="columns")
            current_reults_df.index = [mod_name]
            results_df = pd.concat([results_df, current_reults_df])

        if plot_market:
            # self.original_data["Close"].plot(figsize=(20,5))
            import plotly.graph_objects as go


            fig = go.Figure(data=[go.Candlestick(x=self.original_data.index,
                open=self.original_data['Open'],
                high=self.original_data['High'],
                low=self.original_data['Low'],
                close=self.original_data['Close'])])
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()

        return results_df
    

    def get_equity_plot(self, log_num=0, rolling=0, bidirectional=False):

        # Set equity value and title
        equity = self.log[log_num]["Current_Equity"]
        title = "Equity"

        # If the rolling value is above 0
        if rolling > 0 and bidirectional == False:
            # Make the plotting value a rolling average of n days
            equity = self.log[log_num]["Current_Equity"].rolling(rolling).mean()
            title = f"Equity ({rolling} Day Rolling Avg.)" # Reset the title to be more accurate

        if rolling > 0 and bidirectional == True:
            # Make the plotting value a rolling average of n days
            equity = (self.log[log_num]["Current_Equity"].rolling(rolling).mean() + self.log[log_num].loc[::-1]["Current_Equity"].rolling(rolling).mean()) / 2
            title = f"Equity ({rolling} Day Bidirectional Rolling Avg.)" # Reset the title to be more accurate

        fig, ax = plt.subplots(1,2, figsize=(18,5))

        ax[0].plot(self.original_data["Close"], label="Market", color="#33A2FF") # Could add  / original_data.iloc[0]["Close"] to get percent of original close 
        ax[1].plot(equity, label="Equity", color="#FFC133")
        ax[0].title.set_text("Market")
        ax[1].title.set_text(title)
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)
        plt.show()

        # return self.log[log_num]
    