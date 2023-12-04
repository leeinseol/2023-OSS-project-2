import pandas as pd

def Print_top_10_players(data, years, want_columns) :
    pd.set_option('display.max_colwidth', 20)
    for year in years :
        print("Top 10 players in ", year)
        year_data = pd.DataFrame(index = range(1,11), columns = want_columns)
        for column in want_columns :
            top_10_player = data[data["year"] == year][["batter_name",column]].sort_values(column, ascending = False).head(10)
            get_data_list = []
            for i in range(top_10_player.shape[0]) :
                get_data_list.append((top_10_player.iloc[i,0], top_10_player.iloc[i,1]))
            year_data[column] = get_data_list
        
        print(year_data,"\n")
    pd.reset_option('display.max_colwidth')
    
def Print_highest_war_by_position_in_year(data, year) :
    position_info = data["cp"].unique()
    year_data = data[data["year"] == year]
    print("Year : ", year,"\n")
    for position in position_info :
        if(position == "지명타자") :
            continue
        position_data = year_data[year_data["cp"] == position]
        max_highest_war = position_data["war"].max()
        max_highest_war_player = position_data[position_data["war"] == max_highest_war]["batter_name"].tolist()
        # If there are more player who have same max score, argmax function can't get there name.
        # Because argmax function can get first index that is the max value.
        # So I tyr to get all player who have max score.
        print(max_highest_war_player, " is the highest war player at ", position)
        print("Score is ", max_highest_war,"\n")
        

def Get_correlation_with_salary(data, columns) :
    if "salary" in columns :
        print("Please remove 'salary' in your argument")
    else :
        get_data = pd.concat([data["salary"], data[columns]], axis = 1)
        cor = get_data.corr().iloc[1:,0]
        max_cor = cor.max()
        max_col = cor.index[cor == max_cor].tolist()
        # It is same reason in Print_highest_war_by_position_in_year function
        # It is not common for float types to have the same value. 
        # However, I've written code for this situation.
        for i, column in enumerate(columns) :
            print("Correlation between salary and ", column, " : ", cor[i])
        print("\n",max_col, " has highest correlation with salary")


if __name__=='__main__':
    baseball = pd.read_csv("./2019_kbo_for_kaggle_v2.csv")
    
    Print_top_10_players(baseball, list(range(2015,2019)), ["H","avg","HR","OBP"])
    Print_highest_war_by_position_in_year(baseball, 2018)
    Get_correlation_with_salary(baseball,["R","H","HR","RBI","SB","war","avg","OBP","SLG"] )
