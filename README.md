# screenply

Turn a Screenplay PDF into an analysis-ready Pandas dataframe.

Shout out to https://github.com/pdfminer/pdfminer.six!

## Usage

```python
from screenply.parser import Screenplay

pdf_path = 'Brooklyn_Nine-Nine_1x04_-_M.E._Time.pdf'
screenplay = Screenplay(source=pdf_path, debug_mode=True)
df = screenplay.data

# show top 10 characters, by dialogue length
character_summary = df.pivot_table(
    index='character',
    values='n_chars',
    aggfunc='sum',
).sort_values('n_chars',ascending=False)

character_summary['%'] = character_summary.n_chars/character_summary.n_chars.sum()
print(character_summary.head(10))
```
Result:
```
                n_chars         %
character                        
JAKE               7441  0.345018
AMY                3323  0.154078
CHARLES            2915  0.135160
HOLT               2075  0.096212
ROSSI              1899  0.088051
ROSA               1263  0.058562
TERRY               952  0.044142
MRS. PATTERSON      399  0.018500
SCULLY              359  0.016646
DR. ROSSI           240  0.011128
```
