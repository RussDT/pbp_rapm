import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn import linear_model
import mysql.connector
from collections import defaultdict

# Load position data (if still needed for other purposes)
position_data = pd.read_csv('C:\\Users\\Russell\\Desktop\\28year.csv')
position_dict = dict(zip(position_data['nba_id'], position_data['Position_Number']))

# Load DARKO data
darko_data = pd.read_csv('C:\\Users\\Russell\\Desktop\\DARKO.csv')
d_darko_dict = {}
o_darko_dict = {}
t_darko_dict = {}
for _, row in darko_data.iterrows():
    if 1997 <= row['season'] <= 2024:
        d_darko_dict.setdefault(int(row['nba_id']), {})[int(row['season'])] = row['d_dpm']
        o_darko_dict.setdefault(int(row['nba_id']), {})[int(row['season'])] = row['o_dpm']
        t_darko_dict.setdefault(int(row['nba_id']), {})[int(row['season'])] = row['o_dpm']

def det_def_str(lineup, season):
    d_darko_sum = sum(d_darko_dict.get(int(player), {}).get(season, 0) for player in lineup)
    return 'strong' if d_darko_sum > 2 else 'not_strong'

def det_off_str(lineup, season):
    o_darko_sum = sum(o_darko_dict.get(int(player), {}).get(season, 0) for player in lineup)
    return 'strong' if o_darko_sum > 2.5 else 'not_strong'
def det_tot_str(lineup, season):
    t_darko_sum = sum(t_darko_dict.get(int(player), {}).get(season, 0) for player in lineup)
    return 'strong' if t_darko_sum > 1 else 'not_strong'


def calculate_rapm(data, player_to_col, col_to_player):
    X = lil_matrix((len(data), len(col_to_player)))
    y = np.zeros(len(data))
    counter = 0
    
    player_strength_counter_off = defaultdict(lambda: {'strong': 0, 'total': 0})
    player_strength_counter_def = defaultdict(lambda: {'strong': 0, 'total': 0})
    
    for item in data:
        home_poss, pts, *players, season = item
        all_players = [int(p) for p in players]
        
        if home_poss:
            off_list, def_list = players[5:], players[:5]
        else:
            off_list, def_list = players[:5], players[5:]
        
        opp_def_str = det_tot_str(def_list, int(season))
        opp_off_str = det_tot_str(off_list, int(season))
        
        for p in off_list:
            off_p = f"{p}_off"
            X[counter, player_to_col[off_p]] = 1
            if opp_def_str == 'strong':
                X[counter, player_to_col[f'{p}_facing_strong_off']] = 1
                player_strength_counter_off[int(p)]['strong'] += 1
            player_strength_counter_off[int(p)]['total'] += 1

        for p in def_list:
            def_p = f"{p}_def"
            X[counter, player_to_col[def_p]] = 1
            if opp_off_str == 'strong':
                X[counter, player_to_col[f'{p}_facing_strong_def']] = 1
                player_strength_counter_def[int(p)]['strong'] += 1
            player_strength_counter_def[int(p)]['total'] += 1
            
        y[counter] = pts
        counter += 1

    y -= np.average(y)
    X = X.tocsr()
    clf = linear_model.Ridge(alpha=3000)
    clf.fit(X, y)
    
    dataa = defaultdict(lambda: {'off': 0, 'def': 0,
                                 'facing_strong_off': 0, 'facing_strong_def': 0,
                                 'faced_strong_off': 0, 'faced_strong_def': 0,
                                 'off_possessions': 0, 'def_possessions': 0})
    
    for i, value in enumerate(clf.coef_):
        player_info = col_to_player[i].split('_')
        nba_id = int(player_info[0])
        
        if player_info[1] == 'facing':
            side = player_info[3]
            dataa[nba_id][f"facing_strong_{side}"] = value
        else:
            side = player_info[1]
            dataa[nba_id][side] = value
        
    for nba_id in dataa:
        dataa[nba_id]['faced_strong_off'] = player_strength_counter_off[nba_id]['strong']
        dataa[nba_id]['faced_strong_def'] = player_strength_counter_def[nba_id]['strong']
        dataa[nba_id]['off_possessions'] = player_strength_counter_off[nba_id]['total']
        dataa[nba_id]['def_possessions'] = player_strength_counter_def[nba_id]['total']

    datalist = []
    for ids, player_data in dataa.items():
        if any(player_data[key] is not None for key in player_data):
            temp = [ids] + [player_data[key] for key in ['off', 'def',
                                                         'facing_strong_off', 'facing_strong_def',
                                                         'faced_strong_off', 'faced_strong_def',
                                                         'off_possessions', 'def_possessions']]
            datalist.append(temp)
    
    return datalist

def main():
    # Connect to the database
    db = mysql.connector.connect(
        user='russ_replit2',
        password='RUss4$ratt2',
        host='147.182.232.58',
        database='your_db'
    )
    cur = db.cursor()
    cur.execute("DESCRIBE matchups")
    columns = cur.fetchall()
    
    print("Available fields in the 'matchups' table:")
    for column in columns:
        print(f"- {column[0]} ({column[1]})")
    # Fetch data
    cur.execute("SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5, season FROM matchups WHERE season BETWEEN 1997 AND 2024")
    data = cur.fetchall()
    db.close()
    print('Data fetched')

    all_players = set()
    for item in data:
        all_players.update(item[2:12])

    player_to_col = {}
    col_to_player = {}
    for p in all_players:
        for side in ['off', 'def']:
            p_side = f"{p}_{side}"
            if p_side not in player_to_col:
                number = len(player_to_col)
                player_to_col[p_side] = number
                col_to_player[number] = p_side
        
        for side in ['off', 'def']:
            facin = f"{p}_facing_strong_{side}"
            player_to_col[facin] = len(player_to_col)
            col_to_player[len(col_to_player)] = facin
    
    print('Calculating RAPM')
    datalist = calculate_rapm(data, player_to_col, col_to_player)
    
    # Create DataFrame
    df = pd.DataFrame(datalist, columns=['nba_id', 'off', 'def',
                                         'facing_strong_off', 'facing_strong_def',
                                         'faced_strong_off', 'faced_strong_def',
                                         'off_possessions', 'def_possessions'])

    # Multiply RAPM values by 100
    rapm_columns = ['off', 'def', 'facing_strong_off', 'facing_strong_def']
    df[rapm_columns] = df[rapm_columns].apply(lambda x: x * 100)

    
    # Scale offensive RAPM
    # Scale offensive and defensive RAPM
    for side in ['off', 'def']:
        df[f'strong_perc_{side}'] = df[f'faced_strong_{side}'] / df[f'{side}_possessions']

        # Replace NaN values with 0
        df[f'strong_perc_{side}'] = df[f'strong_perc_{side}'].fillna(0)

        df[f'{side}_vs_strong'] = df[side] + df[f'facing_strong_{side}']
        df[f'{side}_vs_not_strong'] = df[side]

        df[f'overall_{side}'] = (df[f'strong_perc_{side}'] * df[f'{side}_vs_strong'] + 
                                (1 - df[f'strong_perc_{side}']) * df[f'{side}_vs_not_strong'])


    # Now scale the overall metrics
    orapm_target_stdev = 1.48
    drapm_target_stdev = 1.36

    for side in ['off', 'def']:
        # Calculate the weighted average and subtract it
        weighted_avg = np.average(df[f'overall_{side}'], weights=df[f'{side}_possessions'])
        df[f'overall_{side}'] -= weighted_avg

        # Calculate the actual standard deviation of the adjusted overall metric
        actual_stdev = df[f'overall_{side}'].std()
        
        # Set the scaling factor based on the target standard deviation
        if side == 'off':
            scaling_factor = orapm_target_stdev / actual_stdev
        else:
            scaling_factor = drapm_target_stdev / actual_stdev
        
        # Scale overall
        df[f'scaled_overall_{side}'] = df[f'overall_{side}'] * scaling_factor
        
        # Calculate the difference between scaled and original overall values
        df[f'{side}_diff'] = df[f'scaled_overall_{side}'] - df[f'overall_{side}']
        df[f'{side}_diff'] -= weighted_avg
        # Adjust vs_strong and vs_not_strong by the calculated difference
        df[f'scaled_{side}_vs_strong'] = df[f'{side}_vs_strong'] + df[f'{side}_diff']
        df[f'scaled_{side}_vs_not_strong'] = df[f'{side}_vs_not_strong'] + df[f'{side}_diff']

            # Round relevant columns
        columns_to_round = ['off', 'def', 'facing_strong_off', 'facing_strong_def',
                            'off_vs_strong', 'off_vs_not_strong', 'overall_off',
                            'def_vs_strong', 'def_vs_not_strong', 'overall_def']
        df[columns_to_round] = df[columns_to_round].round(2)

        # Merge with player names
    names_df = pd.read_csv('C:\\Users\\Russell\\Desktop\\RAPM\\28year_old.csv')
    df = pd.merge(df, names_df[['nba_id', 'player_name']], on='nba_id')
    df = df[['player_name'] + [col for col in df.columns if col != 'player_name']]

        # Output to CSV
    output_file = 'C:\\Users\\Russell\\Desktop\\python output\\totstrong.csv'
    df.to_csv(output_file, index=False)



if __name__ == '__main__':
    main()