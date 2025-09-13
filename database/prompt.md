 <prompt>
    <context>

      We need to modify the plyr_gm_pass.py script.

      The script needs to add new columns to the database table, and will have to calculate the values for each cell.

    </context>

    <new_database_table_columns>

      Here are the new database columns the script needs to add to the create table statement and insert/upsert functionality

      plyr_gm_pass_cmp_pct DECIMAL(7,4),
      plyr_gm_pass_td_pct DECIMAL (7,4),
      plyr_gm_pass_int_pct DECIMAL (7,4),
      plyr_gm_pass_yds_att DECIMAL (7,4),
      plyr_gm_pass_adj_yds_att DECIMAL (7,4),
      plyr_gm_pass_yds_cmp DECIMAL (7,4),
      plyr_gm_pass_sk_pct DECIMAL (7,4),
      plyr_gm_pass_net_yds_att (7,4),
      plyr_gm_pass_adj_net_yds_att (7,4),

    </new_database_table_columns>

    <calculations>

      Here are the calculations the script needs to perform to determine the values for each new column:

      - plyr_gm_pass_cmp_pct -

      (plyr_gm_pass_cmp) / (plyr_gm_pass_att)

      - plyr_gm_pass_td_pct -
      
      (plyr_gm_pass_td) / (plyr_gm_pass_att)

      - plyr_gm_pass_int_pct - 

      (plyr_gm_pass_int) / (plyr_gm_pass_att)

      - plyr_gm_yds_att - 

      (plyr_gm_pass_yds) / (plyr_gm_pass_att)

      - plyr_gm_pass_adj_yds_att - 

      (plyr_gm_pass_yds + 20 * plyr_gm_pass_td - 45 * plyr_gm_pass_int) / (plyr_gm_pass_att)

      - plyr_gm_pass_sk_pct -

      (plyr_gm_pass_sk) / (plyr_gm_pass_att + plyr_gm_pass_sk)

      - plyr_gm_pass_net_yds_att -

      (plyr_gm_pass_yds - plyr_gm_pass_sk_yds) / (plyr_gm_pass_att + plyr_gm_pass_sk)

      - plyr_gm_pass_adj_net_yds_att - 

      (plyr_gm_pass_yds - plyr_gm_sk_yds + (20 * plyr_gm_pass_td) - (45 * plyr_gm_int)) / (plyr_gm_pass_att + plyr_gm_pass_sk)

    <calculations>

    <agent>

      Once the script is complete, please have @database-quality-checker do a quality control test on the newly calculated data and table columns before considering this complete.

    </agent>



 </prompt>

