import re
import ast

LOG = '''
ors_from_matrix done
[2026-05-26 13:19:04,329: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:04,330: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:04,336: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 5.0, 'concentrating_seconds': 0.0, 'predicted_label': 'neutral', 'confidence': 0.8343116044998169}
[2026-05-26 13:19:04,371: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:04: {'delta': 146.26008246766324, 'theta': 51.79182288974305, 'alpha': 71.71387319895116, 'beta': 1213.453251068727, 'gamma': 247.62310606930703}
[2026-05-26 13:19:04,372: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:04,373: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:146.3→2.9% | theta:51.8→10.4% | alpha:71.7→3.6% | beta:1213.5→12.1% | gamma:247.6→8.3%
[2026-05-26 13:19:04,393: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 83.4%
[2026-05-26 13:19:04,393: WARNING/MainProcess]    20:19:04 | neutral      | 0.83 | 0.60
[2026-05-26 13:19:05,430: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:05,461: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:07,600: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:07,602: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:07,602: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:07,609: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 5.0, 'concentrating_seconds': 0.0, 'predicted_label': 'neutral', 'confidence': 0.7123778462409973}
[2026-05-26 13:19:07,652: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:07: {'delta': 201.50130364062161, 'theta': 73.43815172224548, 'alpha': 43.75796292126925, 'beta': 1210.928068883019, 'gamma': 251.96124366112062}
[2026-05-26 13:19:07,653: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:07,655: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:201.5→4.0% | theta:73.4→14.7% | alpha:43.8→2.2% | beta:1210.9→12.1% | gamma:252.0→8.4%
[2026-05-26 13:19:07,671: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 71.2%
[2026-05-26 13:19:07,672: WARNING/MainProcess]    20:19:07 | neutral      | 0.71 | 0.60
[2026-05-26 13:19:08,703: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:08,739: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:10,447: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:10,448: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:10,449: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:10,453: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 4.0, 'concentrating_seconds': 1.0, 'predicted_label': 'neutral', 'confidence': 0.6985414624214172}
[2026-05-26 13:19:10,482: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:10: {'delta': 381.7833119543916, 'theta': 123.7304674746813, 'alpha': 49.103102154539535, 'beta': 1226.3202913312332, 'gamma': 264.74668665420194}
[2026-05-26 13:19:10,483: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:10,484: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:381.8→7.6% | theta:123.7→24.7% | alpha:49.1→2.5% | beta:1226.3→12.3% | gamma:264.7→8.8%
[2026-05-26 13:19:10,501: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 69.9%
[2026-05-26 13:19:10,501: WARNING/MainProcess]    20:19:10 | neutral      | 0.70 | 0.60
[2026-05-26 13:19:11,535: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:11,555: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:14,386: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:14,388: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:14,389: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:14,394: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 5.0, 'concentrating_seconds': 0.0, 'predicted_label': 'neutral', 'confidence': 0.7688504457473755}
[2026-05-26 13:19:14,427: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:14: {'delta': 219.52074654255065, 'theta': 90.85427757817318, 'alpha': 54.9980734195141, 'beta': 1237.4252179662315, 'gamma': 257.5143423915942}
[2026-05-26 13:19:14,427: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:14,428: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:219.5→4.4% | theta:90.9→18.2% | alpha:55.0→2.7% | beta:1237.4→12.4% | gamma:257.5→8.6%
[2026-05-26 13:19:14,451: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 76.9%
[2026-05-26 13:19:14,453: WARNING/MainProcess]    20:19:14 | neutral      | 0.77 | 0.60
[2026-05-26 13:19:15,478: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:15,498: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:17,261: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:17,263: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:17,263: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:17,269: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 4.0, 'concentrating_seconds': 1.0, 'predicted_label': 'neutral', 'confidence': 0.7313364744186401}
[2026-05-26 13:19:17,312: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:17: {'delta': 107.74816701082462, 'theta': 39.64850239356184, 'alpha': 53.90420741162853, 'beta': 1194.1965013073375, 'gamma': 281.4844550500655}
[2026-05-26 13:19:17,313: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:17,315: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:107.7→2.2% | theta:39.6→7.9% | alpha:53.9→2.7% | beta:1194.2→11.9% | gamma:281.5→9.4%
[2026-05-26 13:19:17,329: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 73.1%
[2026-05-26 13:19:17,331: WARNING/MainProcess]    20:19:17 | neutral      | 0.73 | 0.60
[2026-05-26 13:19:18,374: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:18,424: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:22,214: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:22,217: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:22,217: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:22,223: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 4.5, 'concentrating_seconds': 0.5, 'predicted_label': 'neutral', 'confidence': 0.6479478478431702}
[2026-05-26 13:19:22,257: WARNING/MainProcess] [DEBUG] Brainwave: delta:256.2 theta:67.7 alpha:58.5
[2026-05-26 13:19:22,258: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:22: {'delta': 256.1858596096782, 'theta': 67.68431336724059, 'alpha': 58.505027324930666, 'beta': 1165.2309979090844, 'gamma': 324.9774395804565}
[2026-05-26 13:19:22,260: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:22,261: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:256.2→5.1% | theta:67.7→13.5% | alpha:58.5→2.9% | beta:1165.2→11.7% | gamma:325.0→10.8%
[2026-05-26 13:19:22,276: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 64.8%
[2026-05-26 13:19:22,278: WARNING/MainProcess]    20:19:22 | neutral      | 0.65 | 0.60
[2026-05-26 13:19:23,302: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:23,324: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:26,085: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:26,087: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:26,088: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:26,095: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 4.0, 'concentrating_seconds': 1.0, 'predicted_label': 'neutral', 'confidence': 0.7144556045532227}
[2026-05-26 13:19:26,134: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:26: {'delta': 255.31495969444146, 'theta': 67.05015555072644, 'alpha': 53.24863882067103, 'beta': 1191.5299943170437, 'gamma': 288.5119439215712}
[2026-05-26 13:19:26,135: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:26,136: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:255.3→5.1% | theta:67.1→13.4% | alpha:53.2→2.7% | beta:1191.5→11.9% | gamma:288.5→9.6%
[2026-05-26 13:19:26,154: WARNING/MainProcess] [BROADCAST] Updated live stream: NEUTRAL 71.4%
[2026-05-26 13:19:26,156: WARNING/MainProcess]    20:19:26 | neutral      | 0.71 | 0.60
[2026-05-26 13:19:27,182: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:27,203: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:31,200: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:31,201: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:31,202: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:31,208: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 2.0, 'concentrating_seconds': 3.0, 'predicted_label': 'concentrating', 'confidence': 0.8483617901802063}
[2026-05-26 13:19:31,245: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:31: {'delta': 2603.227519796253, 'theta': 149.22750689210469, 'alpha': 55.70985199755896, 'beta': 860.7779276611165, 'gamma': 219.63231652222578}
[2026-05-26 13:19:31,245: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:31,246: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:2603.2→52.1% | theta:149.2→29.8% | alpha:55.7→2.8% | beta:860.8→8.6% | gamma:219.6→7.3%
[2026-05-26 13:19:31,264: WARNING/MainProcess] [BROADCAST] Updated live stream: CONCENTRATING 84.8%
[2026-05-26 13:19:31,265: WARNING/MainProcess]    20:19:31 | concentrating | 0.85 | 1.00
[2026-05-26 13:19:32,294: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:36,209: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:36,211: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:36,212: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:36,219: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 0.0, 'concentrating_seconds': 5.0, 'predicted_label': 'concentrating', 'confidence': 0.9772180318832397}
[2026-05-26 13:19:36,258: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:36: {'delta': 440.7351693734672, 'theta': 67.40731472452528, 'alpha': 30.90844018033178, 'beta': 314.58801900396753, 'gamma': 123.62552998779465}
[2026-05-26 13:19:36,259: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:36,260: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:440.7→8.8% | theta:67.4→13.5% | alpha:30.9→1.5% | beta:314.6→3.1% | gamma:123.6→4.1%
[2026-05-26 13:19:36,283: WARNING/MainProcess] [BROADCAST] Updated live stream: CONCENTRATING 97.7%
[2026-05-26 13:19:36,284: WARNING/MainProcess]    20:19:36 | concentrating | 0.98 | 1.00
[2026-05-26 13:19:37,355: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:37,407: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:43,259: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:43,261: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:43,262: WARNING/MainProcess] inside preditct #4
[2026-05-26 13:19:43,270: WARNING/MainProcess]   Full result: {'ok': True, 'n_windows': 10, 'total_seconds': 5.0, 'relaxed_seconds': 0.0, 'neutral_seconds': 0.0, 'concentrating_seconds': 5.0, 'predicted_label': 'concentrating', 'confidence': 0.8974868059158325}
[2026-05-26 13:19:43,305: WARNING/MainProcess] [DEBUG] Raw brainwave values at 20:19:43: {'delta': 1135.9212122578635, 'theta': 184.64421935195523, 'alpha': 33.46398200053562, 'beta': 300.4425565612601, 'gamma': 133.55755677127664}
[2026-05-26 13:19:43,307: WARNING/MainProcess] [INFO] Brainwave values changed - new data detected
[2026-05-26 13:19:43,308: WARNING/MainProcess] [SCALING] Raw→Scaled: delta:1135.9→22.7% | theta:184.6→36.9% | alpha:33.5→1.7% | beta:300.4→3.0% | gamma:133.6→4.5%
[2026-05-26 13:19:43,326: WARNING/MainProcess] [BROADCAST] Updated live stream: CONCENTRATING 89.7%
[2026-05-26 13:19:43,328: WARNING/MainProcess]    20:19:43 | concentrating | 0.90 | 1.00
[2026-05-26 13:19:44,364: WARNING/MainProcess] Running Predservice
[2026-05-26 13:19:44,387: WARNING/MainProcess] Input rows size: 1536
[2026-05-26 13:19:50,307: WARNING/MainProcess] generate_feature_vectors_from_matrix done
[2026-05-26 13:19:50,309: WARNING/MainProcess] apply_feature_pipeline done
[2026-05-26 13:19:50,310: WARNING/MainProcess] inside preditct #4
[...truncated for brevity...]
[2026-05-26 13:22:55,429: WARNING/MainProcess] [WAVE AVERAGES] Calculated from data_points - Beta: 0.00Hz, Gamma: 0.00Hz, Alpha: 0.00Hz, Theta: 0.00Hz
'''

RE = re.compile(r"Raw brainwave values at \d{2}:\d{2}:\d{2}: (\{[^}]+\})")

matches = RE.findall(LOG)
if not matches:
    print('No raw brainwave lines found in LOG')
    raise SystemExit(1)

bands = ['delta','theta','alpha','beta','gamma']
S = {b:0.0 for b in bands}
C = {b:0 for b in bands}

for m in matches:
    try:
        d = ast.literal_eval(m)
    except Exception:
        continue
    for b in bands:
        v = d.get(b)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        S[b] += fv
        C[b] += 1

print('Parsed samples:', sum(C.values())//len(bands))
for b in bands:
    if C[b]:
        print(f"{b} avg = {S[b]/C[b]:.6f} ({C[b]} samples)")
    else:
        print(f"{b} avg = None (0 samples)")
