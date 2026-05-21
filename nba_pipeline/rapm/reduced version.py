import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn import linear_model
import mysql.connector
from collections import defaultdict

# Load position data (if still needed for other purposes)

# Load DARKO data
darko_data = pd.read_csv('C:\\Users\\Russell\\Desktop\\DARKO.csv')
d_darko_dict = {}
o_darko_dict = {}
for _, row in darko_data.iterrows():
    if 1997 <= row['season'] <= 2024:
        d_darko_dict.setdefault(int(row['nba_id']), {})[int(row['season'])] = row['d_dpm']
        o_darko_dict.setdefault(int(row['nba_id']), {})[int(row['season'])] = row['o_dpm']

def det_def_str(lineup, season):
    d_darko_sum = sum(d_darko_dict.get(int(player), {}).get(season, 0) for player in lineup)
    if d_darko_sum > 2:
        return 'strong'
    elif d_darko_sum > -1:
        return 'medium'
    else:
        return 'weak'

def det_off_str(lineup, season):
    o_darko_sum = sum(o_darko_dict.get(int(player), {}).get(season, 0) for player in lineup)
    if o_darko_sum > 2.5:
        return 'strong'
    elif o_darko_sum > -1:
        return 'medium'
    else:
        return 'weak'    

def calculate_rapm(data, player_to_col, col_to_player):
    X = lil_matrix((len(data), len(col_to_player)))
    y = np.zeros(len(data))
    counter = 0
    offensive_strength_counter = {'weak': 0, 'medium': 0, 'strong': 0, 'total': 0}
    defensive_strength_counter = {'weak': 0, 'medium': 0, 'strong': 0, 'total': 0}
    
    player_strength_counter_off = defaultdict(lambda: {'weak': 0, 'medium': 0, 'strong': 0})
    player_strength_counter_def = defaultdict(lambda: {'weak': 0, 'medium': 0, 'strong': 0})
    
    for item in data:
        home_poss, pts, *players, season = item
        all_players = [int(p) for p in players]
        
        if home_poss:
            off_list, def_list = players[5:], players[:5]
        else:
            off_list, def_list = players[:5], players[5:]
        
        opp_def_str = det_def_str(def_list, int(season))
        opp_off_str = det_off_str(off_list, int(season))
        
        for p in off_list:
            off_p = f"{p}_off"
            X[counter, player_to_col[off_p]] = 1
            X[counter, player_to_col[f'{p}_facing_{opp_def_str}_off']] = 1
            player_strength_counter_off[int(p)][opp_def_str] += 1

        for p in def_list:
            def_p = f"{p}_def"
            X[counter, player_to_col[def_p]] = 1
            X[counter, player_to_col[f'{p}_facing_{opp_off_str}_def']] = 1
            player_strength_counter_def[int(p)][opp_off_str] += 1

        offensive_strength_counter[opp_def_str] += 1
        defensive_strength_counter[opp_off_str] += 1
        offensive_strength_counter['total'] += 1
        defensive_strength_counter['total'] += 1
            
        y[counter] = pts
        counter += 1

    y -= np.average(y)
    X = X.tocsr()
    clf = linear_model.Ridge(alpha=2000)
    clf.fit(X, y)
    
    dataa = defaultdict(lambda: {'off': None, 'def': None,
                                 'facing_weak_off': None, 'facing_medium_off': None, 'facing_strong_off': None,
                                 'facing_weak_def': None, 'facing_medium_def': None, 'facing_strong_def': None,
                                 'faced_weak_off': 0, 'faced_medium_off': 0, 'faced_strong_off': 0,
                                 'faced_weak_def': 0, 'faced_medium_def': 0, 'faced_strong_def': 0})
    
    for i, value in enumerate(clf.coef_):
        player_info = col_to_player[i].split('_')
        nba_id = int(player_info[0])
        
        if player_info[1] == 'facing':
            strength, side = player_info[2], player_info[3]
            dataa[nba_id][f"facing_{strength}_{side}"] = value
        else:
            side = player_info[1]
            dataa[nba_id][side] = value
        
    for nba_id in dataa:
        dataa[nba_id]['faced_weak_off'] = player_strength_counter_off[nba_id]['weak']
        dataa[nba_id]['faced_medium_off'] = player_strength_counter_off[nba_id]['medium']
        dataa[nba_id]['faced_strong_off'] = player_strength_counter_off[nba_id]['strong']
        dataa[nba_id]['faced_weak_def'] = player_strength_counter_def[nba_id]['weak']
        dataa[nba_id]['faced_medium_def'] = player_strength_counter_def[nba_id]['medium']
        dataa[nba_id]['faced_strong_def'] = player_strength_counter_def[nba_id]['strong']

    datalist = []
    for ids, player_data in dataa.items():
        if any(player_data[key] is not None for key in player_data):
            temp = [ids] + [player_data[key] for key in ['off', 'def',
                                                         'facing_weak_off', 'facing_medium_off', 'facing_strong_off',
                                                         'faced_weak_off', 'faced_medium_off', 'faced_strong_off',
                                                         'facing_weak_def', 'facing_medium_def', 'facing_strong_def',
                                                         'faced_weak_def', 'faced_medium_def', 'faced_strong_def']]
            off_possessions = sum(player_data[f'faced_{strength}_off'] for strength in ['weak', 'medium', 'strong'])
            def_possessions = sum(player_data[f'faced_{strength}_def'] for strength in ['weak', 'medium', 'strong'])
            temp.extend([off_possessions, def_possessions])
            datalist.append(temp)
    
    for strength, count in offensive_strength_counter.items():
        if strength != 'total':
            percentage = (count / offensive_strength_counter['total']) * 100
            print(f"Offensive {strength.capitalize()}: {percentage:.2f}%")
    
    for strength, count in defensive_strength_counter.items():
        if strength != 'total':
            percentage = (count / defensive_strength_counter['total']) * 100
            print(f"Defensive {strength.capitalize()}: {percentage:.2f}%")
    
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
        
        for v in ['facing_weak', 'facing_medium', 'facing_strong']:
            for side in ['off', 'def']:
                facin = f"{p}_{v}_{side}"
                player_to_col[facin] = len(player_to_col)
                col_to_player[len(col_to_player)] = facin
    
    print('Calculating RAPM')
    datalist = calculate_rapm(data, player_to_col, col_to_player)
    
    # Create DataFrame
    df = pd.DataFrame(datalist, columns=['nba_id', 'off', 'def',
                                         'facing_weak_off', 'facing_medium_off', 'facing_strong_off',
                                         'faced_weak_off', 'faced_medium_off', 'faced_strong_off',
                                         'facing_weak_def', 'facing_medium_def', 'facing_strong_def',
                                         'faced_weak_def', 'faced_medium_def', 'faced_strong_def',
                                         'off_possessions', 'def_possessions'])

    # Multiply RAPM values by 100
    rapm_columns = ['off', 'def', 'facing_weak_off', 'facing_medium_off', 'facing_strong_off',
                    'facing_weak_def', 'facing_medium_def', 'facing_strong_def']
    df[rapm_columns] = df[rapm_columns].apply(lambda x: x * 100)

    weighted_avg_off = np.average(df['off'], weights=df['off_possessions'])
    df['off'] -= weighted_avg_off

    # Scale defensive RAPM
    weighted_avg_def = np.average(df['def'], weights=df['def_possessions'])
    df['def'] -= weighted_avg_def

    # Scale facing offensive RAPM
    for strength in ['weak', 'medium', 'strong']:
        weighted_avg = np.average(df[f'facing_{strength}_off'], weights=df[f'faced_{strength}_off'])
        df[f'facing_{strength}_off'] -= weighted_avg

    # Scale facing defensive RAPM
    for strength in ['weak', 'medium', 'strong']:
        weighted_avg = np.average(df[f'facing_{strength}_def'], weights=df[f'faced_{strength}_def'])
        df[f'facing_{strength}_def'] -= weighted_avg
    # Calculate percentages and overall metrics
    for side in ['off', 'def']:
        for strength in ['weak', 'medium', 'strong']:
            df[f'{strength}_perc_{side}'] = df[f'faced_{strength}_{side}'] / df[f'{side}_possessions']
            df[f'{side}_vs_{strength}'] = df[side] + df[f'facing_{strength}_{side}']
        
        df[f'overall_{side}'] = sum(df[f'{strength}_perc_{side}'] * df[f'{side}_vs_{strength}'] for strength in ['weak', 'medium', 'strong'])

    # Round relevant columns
    columns_to_round = ['off', 'def', 'facing_weak_off', 'facing_medium_off', 'facing_strong_off',
                        'facing_weak_def', 'facing_medium_def', 'facing_strong_def',
                        'off_vs_weak', 'off_vs_medium', 'off_vs_strong', 'overall_off',
                        'def_vs_weak', 'def_vs_medium', 'def_vs_strong', 'overall_def']
    df[columns_to_round] = df[columns_to_round].round(2)

    # Merge with player names
    names_df = pd.read_csv('C:\\Users\\Russell\\Desktop\\RAPM\\28year_old.csv')
    df = pd.merge(df, names_df[['nba_id', 'player_name']], on='nba_id')
    df = df[['player_name'] + [col for col in df.columns if col != 'player_name']]

    # Output to CSV
    output_file = 'C:\\Users\\Russell\\Desktop\\python output\\simplified_scaled_facing_balanced_strength_rapm.csv'
    df.to_csv(output_file, index=False)

    print(df[df['nba_id'] == 203076])

if __name__ == '__main__':
    main()