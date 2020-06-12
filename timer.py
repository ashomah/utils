timer = pd.DataFrame(columns=['pipeline', 'section', 'step', 'start', 'end', 'total'])

def start_timer(pipeline, section, step):
    global running_step
    timer_start = datetime.now()
    running_step = {'pipeline': pipeline, 'section': section, 'step': step, 'start': timer_start, 'end': np.nan, 'total': np.nan}
        
def stop_timer():
    global running_step
    global timer
    timer_end = datetime.now()
    running_step['end']   = timer_end
    running_step['total'] = (running_step['end'] - running_step['start'])
    timer = timer.append(running_step, ignore_index=True)

def timer_summary():
    global timer
    if len(timer) == 0:
        print('NO DATA: The Timer is empty.')
    else:
        plot_timer = pd.DataFrame()
        plot_timer['pipeline-section-step']  = timer.apply(lambda row: row.step+' | '+row.section+' | '+row.pipeline, axis=1)
        plot_timer['total']         = timer.total
        plot_timer['total_display'] = timer.apply(lambda row: '{:.0f}min {:0>9.6f}s'.format(row.total.total_seconds()//60, row.total.total_seconds()%60), axis=1)
    #     display(plot_timer)
        
        print('Total Time: {:.0f}min {:0>9.6f}s'.format(timer.loc[:,'total'].sum().total_seconds()//60, timer.loc[:,'total'].sum().total_seconds()%60))
        print('Start Time: {}'.format(timer.loc[:,'start'].min()))
        print('End Time:   {}'.format(timer.loc[:,'end'].max()))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x = plot_timer['pipeline-section-step'], y = plot_timer.total, text=plot_timer.total_display, hoverinfo='text'))

        fig.update_layout(title='Execution Times', plot_bgcolor=ks_white, hovermode='x',
#                           autosize=False, width=500,
                          height=650, margin=go.layout.Margin(b=350))
        fig.update_traces(marker_color = ks_red)
        fig.update_xaxes(tickfont=dict(size=10), tickangle=270,
                         showline=True, linecolor=ks_red, showgrid=False, gridcolor=ks_darkgrey,
                         automargin=True)
        fig.update_yaxes(title_text='NanoSeconds', title_font=dict(size=10), tickfont=dict(size=10),
                         showline=True, zeroline=True, zerolinecolor=ks_darkgray, automargin=True,
                         linecolor=ks_red, showgrid=True, gridcolor=ks_darkgrey)

        fig.show()
