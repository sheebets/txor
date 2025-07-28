import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import numpy as np

st.title('TxOdds Reader')
st.text('Filter and visualize betting data with RECDATE vs PRICES scatter plot')

# File uploader
upload_file = st.file_uploader('Upload your CSV file', type=['csv'])

if upload_file:
    # Load user's data
    df = pd.read_csv(upload_file)
    
    st.header('Data Overview')
    st.write(f"Total records: {len(df)}")
    
    # Condensed column names display
    st.write("**Columns:**", ", ".join(df.columns.tolist()))
    
    # Show first values of key columns as description
    st.subheader('Data Sample Information')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write("**SPORT:**")
        st.write(f"{df['SPORT'].iloc[0] if 'SPORT' in df.columns and not df['SPORT'].empty else 'N/A'}")
    
    with col2:
        st.write("**FIXTUREID:**")
        st.write(f"{df['FIXTUREID'].iloc[0] if 'FIXTUREID' in df.columns and not df['FIXTUREID'].empty else 'N/A'}")
    
    with col3:
        st.write("**MATCH_DATE:**")
        st.write(f"{df['MATCH_DATE'].iloc[0] if 'MATCH_DATE' in df.columns and not df['MATCH_DATE'].empty else 'N/A'}")
    
    with col4:
        st.write("**PARTICIPANT1:**")
        st.write(f"{df['PARTICIPANT1'].iloc[0] if 'PARTICIPANT1' in df.columns and not df['PARTICIPANT1'].empty else 'N/A'}")
    
    with col5:
        st.write("**PARTICIPANT2:**")
        st.write(f"{df['PARTICIPANT2'].iloc[0] if 'PARTICIPANT2' in df.columns and not df['PARTICIPANT2'].empty else 'N/A'}")
    
    # Show sample data
    with st.expander("View Sample Data"):
        st.write(df.head())
    
    # Create filters in sidebar (always shown)
    st.sidebar.header('Filters')
    
    # Enhanced Date & Time filter
    st.sidebar.subheader('📅 Date & Time Filter')
    
    # Initialize variables
    datetime_range = None
    start_datetime = None
    end_datetime = None
    
    # Convert RECDATE to datetime for filtering
    if 'RECDATE' in df.columns:
        try:
            df['RECDATE_temp'] = pd.to_datetime(df['RECDATE'])
            
            # Get min and max datetime from data
            min_datetime = df['RECDATE_temp'].min()
            max_datetime = df['RECDATE_temp'].max()
            
            # Extract date range
            min_date = min_datetime.date()
            max_date = max_datetime.date()
            
            # Date range picker using calendar interface
            st.sidebar.write("**Select Date Range:**")
            
            # Create two columns for start and end date
            date_col1, date_col2 = st.sidebar.columns(2)
            
            with date_col1:
                start_date = st.date_input(
                    'Start Date',
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key='start_date'
                )
            
            with date_col2:
                end_date = st.date_input(
                    'End Date',
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key='end_date'
                )
            
            # Validate date range
            if start_date > end_date:
                st.sidebar.error("❌ Start date must be before or equal to end date!")
                start_date = end_date
            
            # Time range picker
            st.sidebar.write("**Select Time Range:**")
            
            # Create two columns for start and end time
            time_col1, time_col2 = st.sidebar.columns(2)
            
            with time_col1:
                start_time = st.time_input(
                    'Start Time',
                    value=time(0, 0),
                    key='start_time'
                )
            
            with time_col2:
                end_time = st.time_input(
                    'End Time',
                    value=time(23, 59),
                    key='end_time'
                )
            
            # Combine date and time
            start_datetime = pd.Timestamp.combine(start_date, start_time)
            end_datetime = pd.Timestamp.combine(end_date, end_time)
            
            # Display selected range with better formatting
            st.sidebar.markdown("**📋 Selected Range:**")
            st.sidebar.success(f"**From:** {start_datetime.strftime('%Y-%m-%d %H:%M')}")
            st.sidebar.success(f"**To:** {end_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            # Show data distribution in selected range
            filtered_by_date = df[(df['RECDATE_temp'] >= start_datetime) & (df['RECDATE_temp'] <= end_datetime)]
            st.sidebar.info(f"📊 Records in range: **{len(filtered_by_date):,}** / {len(df):,}")
            
            # Quick date range buttons
            st.sidebar.write("**Quick Select:**")
            quick_col1, quick_col2 = st.sidebar.columns(2)
            
            with quick_col1:
                if st.button("📅 Today", key='today'):
                    today = datetime.now().date()
                    st.session_state.start_date = today
                    st.session_state.end_date = today
                    st.rerun()
                
                if st.button("📅 Last 7 Days", key='last_7_days'):
                    today = datetime.now().date()
                    week_ago = today - pd.Timedelta(days=7)
                    st.session_state.start_date = max(week_ago.date(), min_date)
                    st.session_state.end_date = min(today, max_date)
                    st.rerun()
            
            with quick_col2:
                if st.button("📅 This Month", key='this_month'):
                    today = datetime.now().date()
                    month_start = today.replace(day=1)
                    st.session_state.start_date = max(month_start, min_date)
                    st.session_state.end_date = min(today, max_date)
                    st.rerun()
                
                if st.button("📅 All Data", key='all_data'):
                    st.session_state.start_date = min_date
                    st.session_state.end_date = max_date
                    st.rerun()
            
            datetime_range = (start_datetime, end_datetime)
            
        except Exception as e:
            st.sidebar.error(f"❌ Could not parse dates: {str(e)}")
            datetime_range = None
            start_datetime = None
            end_datetime = None
            df['RECDATE_temp'] = None
    else:
        st.sidebar.warning("⚠️ RECDATE column not found - date filtering disabled")
        datetime_range = None
        start_datetime = None
        end_datetime = None
        df['RECDATE_temp'] = None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader('🎯 Other Filters')
    
    # Check which columns exist and create filters accordingly
    required_columns = ['SUPERODDSTYPE', 'MARKETPARAMETERS', 'MARKETPERIOD', 'INRUNNING', 'BOOKMAKER', 'PRICES', 'RECDATE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.sidebar.error(f"❌ Missing required columns: {missing_columns}")
        st.stop()
    
    # SUPERODDSTYPE filter (replaces SOT)
    superodds_options = ['All'] + sorted(df['SUPERODDSTYPE'].dropna().unique().tolist())
    selected_superodds = st.sidebar.selectbox('🎯 Select SUPERODDSTYPE:', superodds_options)
    
    # Apply SUPERODDSTYPE filter first to narrow down Market Parameters options
    temp_filtered_df = df.copy()
    if selected_superodds != 'All':
        temp_filtered_df = temp_filtered_df[temp_filtered_df['SUPERODDSTYPE'] == selected_superodds]
    
    # Split MARKETPARAMETERS by comma and create separate components
    def split_market_parameters(market_param):
        """Split market parameters by comma and return components"""
        if pd.isna(market_param):
            return []
        return [part.strip() for part in str(market_param).split(',')]
    
    # Extract market parameter components
    temp_filtered_df['MARKET_COMPONENTS'] = temp_filtered_df['MARKETPARAMETERS'].apply(split_market_parameters)
    
    # Create separate columns for each component position
    max_components = temp_filtered_df['MARKET_COMPONENTS'].apply(len).max() if len(temp_filtered_df) > 0 else 0
    
    for i in range(max_components):
        temp_filtered_df[f'COMPONENT_{i+1}'] = temp_filtered_df['MARKET_COMPONENTS'].apply(
            lambda x: x[i] if len(x) > i else None
        )
    
    # Search box for Market Parameters
    search_term = st.sidebar.text_input('🔍 Search Market Parameters:', placeholder='Type to search...')
    
    # Apply search filter to temp data if search term exists
    if search_term:
        temp_filtered_df = temp_filtered_df[
            temp_filtered_df['MARKETPARAMETERS'].str.contains(search_term, case=False, na=False)
        ]
    
    # Create filters for all available market components (single selection)
    # Order: Market Parameter 2 first, then Market Parameter 1, then others
    component_selections = {}
    
    # Handle Market Parameter 2 first (if it exists)
    if max_components >= 2:
        col_name = 'COMPONENT_2'
        if col_name in temp_filtered_df.columns:
            available_options = sorted(temp_filtered_df[col_name].dropna().unique().tolist())
            if len(available_options) > 0:
                options_with_all = ['All'] + available_options
                selected_option = st.sidebar.selectbox(
                    '📊 Market Parameter 2:', 
                    options_with_all,
                    index=0,  # Default to "All"
                    key='comp_2'
                )
                st.sidebar.write(f"Available options: {len(available_options)}")
                component_selections[col_name] = selected_option
                
                # Apply this component filter to temp data for cascading
                if selected_option != 'All':
                    temp_filtered_df = temp_filtered_df[temp_filtered_df[col_name] == selected_option]
    
    # Handle Market Parameter 1 second
    if max_components >= 1:
        col_name = 'COMPONENT_1'
        if col_name in temp_filtered_df.columns:
            available_options = sorted(temp_filtered_df[col_name].dropna().unique().tolist())
            if len(available_options) > 0:
                options_with_all = ['All'] + available_options
                selected_option = st.sidebar.selectbox(
                    '📊 Market Parameter 1:', 
                    options_with_all,
                    index=0,  # Default to "All"
                    key='comp_1'
                )
                st.sidebar.write(f"Available options: {len(available_options)}")
                component_selections[col_name] = selected_option
                
                # Apply this component filter to temp data for cascading
                if selected_option != 'All':
                    temp_filtered_df = temp_filtered_df[temp_filtered_df[col_name] == selected_option]
    
    # Handle remaining components (3, 4, 5, etc.) in order
    for i in range(3, max_components + 1):
        col_name = f'COMPONENT_{i}'
        if col_name in temp_filtered_df.columns:
            available_options = sorted(temp_filtered_df[col_name].dropna().unique().tolist())
            if len(available_options) > 0:
                options_with_all = ['All'] + available_options
                selected_option = st.sidebar.selectbox(
                    f'📊 Market Parameter {i}:', 
                    options_with_all,
                    index=0,  # Default to "All"
                    key=f'comp_{i}'
                )
                st.sidebar.write(f"Available options: {len(available_options)}")
                component_selections[col_name] = selected_option
                
                # Apply this component filter to temp data for cascading
                if selected_option != 'All':
                    temp_filtered_df = temp_filtered_df[temp_filtered_df[col_name] == selected_option]
    
    # MARKETPARAMETERS filter with dynamic options based on component selections
    available_markets = sorted(temp_filtered_df['MARKETPARAMETERS'].dropna().unique().tolist())
    
    # Market Parameters selectbox with filtered options
    market_options = ['All'] + available_markets
    selected_market = st.sidebar.selectbox('📋 Select Market Parameters:', market_options)
    
    # Apply market parameter filter
    if selected_market != 'All':
        temp_filtered_df = temp_filtered_df[temp_filtered_df['MARKETPARAMETERS'] == selected_market]
    
    # MARKETPERIOD filter (multiselect)
    available_periods = sorted(temp_filtered_df['MARKETPERIOD'].dropna().unique().tolist())
    selected_periods = st.sidebar.multiselect(
        '⏰ Market Period:', 
        available_periods,
        default=available_periods,  # Default is ALL (select all available options)
        key='periods'
    )
    st.sidebar.write(f"Available Market Periods: {len(available_periods)}")
    
    # Apply market period filter
    if selected_periods:  # If something is selected, filter by it
        temp_filtered_df = temp_filtered_df[temp_filtered_df['MARKETPERIOD'].isin(selected_periods)]
    
    # INRUNNING filter (default to 1)
    available_inrunning = sorted(temp_filtered_df['INRUNNING'].dropna().unique().tolist())
    inrunning_options = available_inrunning
    
    # Set default to 1 if it exists, otherwise first available option
    default_inrunning = 1 if 1 in available_inrunning else (available_inrunning[0] if available_inrunning else None)
    
    if available_inrunning:
        selected_inrunning = st.sidebar.selectbox(
            '🏃 Select In-Running:', 
            inrunning_options,
            index=available_inrunning.index(default_inrunning) if default_inrunning in available_inrunning else 0
        )
        
        # Apply inrunning filter
        temp_filtered_df = temp_filtered_df[temp_filtered_df['INRUNNING'] == selected_inrunning]
    else:
        selected_inrunning = None
    
    # BOOKMAKER filter (multiselect)
    available_bookmakers = sorted(temp_filtered_df['BOOKMAKER'].dropna().unique().tolist())
    st.sidebar.write(f"Available Bookmakers: {len(available_bookmakers)}")
    
    selected_bookmakers = st.sidebar.multiselect(
        '🏪 Select Bookmakers:', 
        available_bookmakers, 
        default=available_bookmakers[:5] if len(available_bookmakers) > 5 else available_bookmakers
    )
    
    # Apply bookmaker filter to temp data for price range calculation
    if selected_bookmakers and len(selected_bookmakers) < len(available_bookmakers):
        temp_filtered_df = temp_filtered_df[temp_filtered_df['BOOKMAKER'].isin(selected_bookmakers)]
    
    # PRICES filter (slider range)
    st.sidebar.subheader('💰 Price Range Filter')
    
    # Parse prices to get numeric values for filtering
    def parse_prices_for_filter(price_str):
        try:
            if pd.isna(price_str):
                return np.nan
            return float(price_str)
        except:
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', str(price_str))
                if numbers:
                    return float(numbers[0])
                return np.nan
            except:
                return np.nan
    
    # Calculate price range from filtered data
    temp_filtered_df['PRICES_numeric_temp'] = temp_filtered_df['PRICES'].apply(parse_prices_for_filter)
    price_data = temp_filtered_df['PRICES_numeric_temp'].dropna()
    
    if len(price_data) > 0:
        min_price = float(price_data.min())
        max_price = float(price_data.max())
        
        # Create price range slider
        if min_price < max_price:
            price_range = st.sidebar.slider(
                '💰 Select Price Range:',
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=0.01 if max_price - min_price < 10 else 0.1,
                format="%.2f"
            )
            
            min_selected_price, max_selected_price = price_range
            
            # Display selected price range
            st.sidebar.success(f"**Selected Range:** {min_selected_price:.2f} - {max_selected_price:.2f}")
            
            # Show how many records are in the price range
            price_filtered_count = len(temp_filtered_df[
                (temp_filtered_df['PRICES_numeric_temp'] >= min_selected_price) & 
                (temp_filtered_df['PRICES_numeric_temp'] <= max_selected_price)
            ])
            st.sidebar.info(f"📊 Records in price range: **{price_filtered_count:,}** / {len(temp_filtered_df):,}")
            
        else:
            st.sidebar.info(f"💰 Single price value: {min_price:.2f}")
            price_range = (min_price, max_price)
            min_selected_price, max_selected_price = price_range
    else:
        st.sidebar.warning("⚠️ No valid price data found for filtering")
        price_range = None
        min_selected_price = None
        max_selected_price = None
    
    # RUN button to apply all filters
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    
    # Only proceed with analysis if RUN button is clicked
    if not run_analysis:
        st.info("👆 Configure your filters above and click 'RUN ANALYSIS' to generate charts and data.")
        st.stop()
    
    # Apply all filters to get final filtered data
    filtered_df = df.copy()
    
    # Split market parameters for main dataset too
    filtered_df['MARKET_COMPONENTS'] = filtered_df['MARKETPARAMETERS'].apply(split_market_parameters)
    
    # Create component columns for main dataset
    for i in range(max_components):
        filtered_df[f'COMPONENT_{i+1}'] = filtered_df['MARKET_COMPONENTS'].apply(
            lambda x: x[i] if len(x) > i else None
        )
    
    # Apply date filter only if RECDATE exists
    if datetime_range and start_datetime and end_datetime and 'RECDATE' in df.columns:
        try:
            filtered_df = filtered_df[
                (filtered_df['RECDATE_temp'] >= start_datetime) & 
                (filtered_df['RECDATE_temp'] <= end_datetime)
            ]
            st.write(f"📅 Date & Time filter applied: {len(filtered_df):,} records between {start_datetime.strftime('%Y-%m-%d %H:%M')} and {end_datetime.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            st.warning(f"Date filtering error: {str(e)}")
    
    # Apply SUPERODDSTYPE filter
    if selected_superodds != 'All':
        filtered_df = filtered_df[filtered_df['SUPERODDSTYPE'] == selected_superodds]
        st.write(f"🎯 SUPERODDSTYPE filter applied: {len(filtered_df):,} records match '{selected_superodds}'")
    
    # Apply search term filter
    if search_term:
        filtered_df = filtered_df[
            filtered_df['MARKETPARAMETERS'].str.contains(search_term, case=False, na=False)
        ]
        st.write(f"🔍 Search filter applied: {len(filtered_df):,} records contain '{search_term}'")
    
    # Apply all component filters with detailed feedback
    for i, (col_name, selected_option) in enumerate(component_selections.items(), 1):
        if selected_option != 'All':
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[col_name] == selected_option]
            st.write(f"📊 Market Parameter {i} filter applied: {len(filtered_df):,} records match '{selected_option}' (reduced from {before_count:,})")
    
    # Apply market parameter filter (full string)
    if selected_market != 'All':
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['MARKETPARAMETERS'] == selected_market]
        st.write(f"📋 Market Parameters filter applied: {len(filtered_df):,} records match (reduced from {before_count:,})")
    
    # Apply market period filter
    if selected_periods and len(selected_periods) < len(available_periods):
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['MARKETPERIOD'].isin(selected_periods)]
        st.write(f"⏰ Market Period filter applied: {len(filtered_df):,} records match selected periods (reduced from {before_count:,})")
    
    # Apply inrunning filter
    if selected_inrunning is not None:
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['INRUNNING'] == selected_inrunning]
        st.write(f"🏃 In-Running filter applied: {len(filtered_df):,} records match '{selected_inrunning}' (reduced from {before_count:,})")
    
    # Apply bookmaker filter
    if selected_bookmakers and len(selected_bookmakers) < len(available_bookmakers):
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['BOOKMAKER'].isin(selected_bookmakers)]
        st.write(f"🏪 Bookmaker filter applied: {len(filtered_df):,} records match selected bookmakers (reduced from {before_count:,})")
    
    # Apply price range filter
    if price_range and min_selected_price is not None and max_selected_price is not None:
        # Add numeric prices column to main filtered data
        filtered_df['PRICES_numeric_temp'] = filtered_df['PRICES'].apply(parse_prices_for_filter)
        
        before_count = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df['PRICES_numeric_temp'] >= min_selected_price) & 
            (filtered_df['PRICES_numeric_temp'] <= max_selected_price)
        ]
        st.write(f"💰 Price range filter applied: {len(filtered_df):,} records between {min_selected_price:.2f} - {max_selected_price:.2f} (reduced from {before_count:,})")
        
        # Clean up temporary column
        filtered_df = filtered_df.drop('PRICES_numeric_temp', axis=1)
    
    # Force refresh of filtered data to ensure all subsequent operations use filtered data
    filtered_df = filtered_df.copy()
    
    st.markdown("---")
    st.header('📊 Analysis Results')
    st.write(f"**Final filtered dataset: {len(filtered_df):,} records**")
    
    if len(filtered_df) > 0:
        # Data preprocessing for plotting - use the ALREADY FILTERED data
        plot_df = filtered_df.copy()  # This now includes all filters including date/time
        
        # Convert RECDATE to datetime (handle various formats)
        try:
            plot_df['RECDATE_parsed'] = pd.to_datetime(plot_df['RECDATE'])
        except:
            st.warning("Could not parse all RECDATE values. Using original values.")
            plot_df['RECDATE_parsed'] = plot_df['RECDATE']
        
        # Handle PRICES - convert to numeric if possible
        def parse_prices(price_str):
            try:
                # If it's already a number
                if pd.isna(price_str):
                    return np.nan
                
                # Try to convert directly to float
                return float(price_str)
            except:
                try:
                    # Try to extract first number from string if it contains multiple values
                    import re
                    numbers = re.findall(r'\d+\.?\d*', str(price_str))
                    if numbers:
                        return float(numbers[0])
                    return np.nan
                except:
                    return np.nan
        
        plot_df['PRICES_numeric'] = plot_df['PRICES'].apply(parse_prices)
        
        # Remove rows where we couldn't parse prices - this preserves all other filtering
        plot_df_clean = plot_df.dropna(subset=['PRICES_numeric'])
        
        # Show confirmation that plots use filtered data
        st.write(f"**📊 Charts will display {len(plot_df_clean):,} filtered records**")
        
        if len(plot_df_clean) > 0:
            # Get unique bookmakers and create color map for all plots
            unique_bookmakers = plot_df_clean['BOOKMAKER'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_bookmakers)))
            color_map = dict(zip(unique_bookmakers, colors))
            
            # Calculate date range for consistent formatting across all plots
            date_range_span = plot_df_clean['RECDATE_parsed'].max() - plot_df_clean['RECDATE_parsed'].min()
            
            st.write(f"**🎯 Plotting data from {plot_df_clean['RECDATE_parsed'].min().strftime('%Y-%m-%d %H:%M')} to {plot_df_clean['RECDATE_parsed'].max().strftime('%Y-%m-%d %H:%M')}**")
            
            # First scatter plot: RECDATE vs O1 (OVER)
            if 'O1' in plot_df_clean.columns:
                st.header('Scatter Plot: RECDATE vs O1 (OVER) - Color-coded by Bookmaker')
                
                # Create the first plot
                fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
                
                # Filter out rows where O1 is not numeric or is NaN
                plot_df_o1 = plot_df_clean.dropna(subset=['O1']).copy()
                
                if len(plot_df_o1) > 0:
                    # Create scatter plot for each bookmaker (X = RECDATE, Y = O1)
                    for bookmaker in unique_bookmakers:
                        if bookmaker in plot_df_o1['BOOKMAKER'].values:
                            bookmaker_data = plot_df_o1[plot_df_o1['BOOKMAKER'] == bookmaker]
                            ax1.scatter(x=bookmaker_data['RECDATE_parsed'], 
                                       y=bookmaker_data['O1'],
                                       alpha=0.7, 
                                       c=[color_map[bookmaker]], 
                                       label=bookmaker,
                                       s=50)
                    
                    ax1.set_xlabel('RECDATE')
                    ax1.set_ylabel('O1 (OVER)')
                    ax1.set_title(f'RECDATE vs O1 (OVER) by Bookmaker (n={len(plot_df_o1):,})')
                    
                    # Add legend
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Format x-axis to show dates properly
                    if date_range_span.days <= 1:
                        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                    elif date_range_span.days <= 7:
                        ax1.xaxis.set_major_locator(mdates.DayLocator())
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
                    elif date_range_span.days <= 30:
                        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        ax1.xaxis.set_minor_locator(mdates.DayLocator())
                    else:
                        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        ax1.xaxis.set_minor_locator(mdates.DayLocator())
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig1)
                else:
                    st.warning("No valid O1 (OVER) data found for plotting.")
            else:
                st.warning("O1 column not found in data.")
            
            # Second scatter plot: RECDATE vs O2 (UNDER)
            if 'O2' in plot_df_clean.columns:
                st.header('Scatter Plot: RECDATE vs O2 (UNDER) - Color-coded by Bookmaker')
                
                # Create the second plot
                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
                
                # Filter out rows where O2 is not numeric or is NaN
                plot_df_o2 = plot_df_clean.dropna(subset=['O2']).copy()
                
                if len(plot_df_o2) > 0:
                    # Create scatter plot for each bookmaker (X = RECDATE, Y = O2)
                    for bookmaker in unique_bookmakers:
                        if bookmaker in plot_df_o2['BOOKMAKER'].values:
                            bookmaker_data = plot_df_o2[plot_df_o2['BOOKMAKER'] == bookmaker]
                            ax2.scatter(x=bookmaker_data['RECDATE_parsed'], 
                                       y=bookmaker_data['O2'],
                                       alpha=0.7, 
                                       c=[color_map[bookmaker]], 
                                       label=bookmaker,
                                       s=50)
                    
                    ax2.set_xlabel('RECDATE')
                    ax2.set_ylabel('O2 (UNDER)')
                    ax2.set_title(f'RECDATE vs O2 (UNDER) by Bookmaker (n={len(plot_df_o2):,})')
                    
                    # Add legend
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Format x-axis to show dates properly
                    if date_range_span.days <= 1:
                        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                    elif date_range_span.days <= 7:
                        ax2.xaxis.set_major_locator(mdates.DayLocator())
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                        ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
                    elif date_range_span.days <= 30:
                        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        ax2.xaxis.set_minor_locator(mdates.DayLocator())
                    else:
                        ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        ax2.xaxis.set_minor_locator(mdates.DayLocator())
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig2)
                else:
                    st.warning("No valid O2 (UNDER) data found for plotting.")
            else:
                st.warning("O2 column not found in data.")
            
            # Third scatter plot: RECDATE vs MARKET PARAMETERS
            st.header('Scatter Plot: RECDATE vs MARKET PARAMETERS (Color-coded by Bookmaker)')
            
            # Create the third plot
            fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))  # Larger figure for better label readability
            
            # Get unique market parameters and assign numeric values
            unique_markets = plot_df_clean['MARKETPARAMETERS'].unique()
            market_to_num = {market: i for i, market in enumerate(unique_markets)}
            plot_df_clean['MARKETPARAMETERS_numeric'] = plot_df_clean['MARKETPARAMETERS'].map(market_to_num)
            
            # Create scatter plot for each bookmaker (X = RECDATE, Y = MARKET PARAMETERS)
            for bookmaker in unique_bookmakers:
                bookmaker_data = plot_df_clean[plot_df_clean['BOOKMAKER'] == bookmaker]
                ax3.scatter(x=bookmaker_data['RECDATE_parsed'], 
                           y=bookmaker_data['MARKETPARAMETERS_numeric'],
                           alpha=0.7, 
                           c=[color_map[bookmaker]], 
                           label=bookmaker,
                           s=50)
            
            ax3.set_xlabel('RECDATE')
            ax3.set_ylabel('MARKET PARAMETERS')
            ax3.set_title(f'RECDATE vs MARKET PARAMETERS by Bookmaker (n={len(plot_df_clean):,})')
            
            # Set y-axis labels to show complete market parameter names
            ax3.set_yticks(range(len(unique_markets)))
            # Show full market parameter names (no truncation)
            ax3.set_yticklabels(unique_markets, rotation=0, ha='right', fontsize=8)
            
            # Add legend
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Format x-axis to show dates properly
            if date_range_span.days <= 1:
                ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
            elif date_range_span.days <= 7:
                ax3.xaxis.set_major_locator(mdates.DayLocator())
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax3.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            elif date_range_span.days <= 30:
                ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax3.xaxis.set_minor_locator(mdates.DayLocator())
            else:
                ax3.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax3.xaxis.set_minor_locator(mdates.DayLocator())
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig3)
            
            # Show Filtered Data Sheet
            st.header('📋 Filtered Data Sheet')
            
            # Select specific columns requested
            sheet_columns = ['SUPERODDSTYPE', 'BOOKMAKER', 'MARKETPARAMETERS', 'RECDATE', 'PRICES']
            
            # Create display dataframe with selected columns
            display_df = plot_df_clean[sheet_columns].copy()
            
            # Format RECDATE for better display
            display_df['RECDATE'] = display_df['RECDATE'].astype(str)
            
            # Rename MARKETPARAMETERS to MARKET PARAMETER for display
            display_df = display_df.rename(columns={'MARKETPARAMETERS': 'MARKET PARAMETER'})
            
            # Show data based on number of records
            if len(display_df) <= 100:
                st.write(f"**Showing all {len(display_df):,} filtered records:**")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
            else:
                st.write(f"**Showing first 100 of {len(display_df):,} filtered records:**")
                st.dataframe(
                    display_df.head(100),
                    use_container_width=True,
                    height=400
                )
            
            # Add download button for full filtered data
            if len(display_df) > 0:
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_data,
                    file_name=f"filtered_betting_data_{len(display_df)}_records.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Show summary statistics
            st.header('📊 Summary Statistics')
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('PRICES Statistics')
                st.write(plot_df_clean['PRICES_numeric'].describe())
            
            with col2:
                st.subheader('Data Summary')
                summary_data = {
                    'Total Records': f"{len(plot_df_clean):,}",
                    'Unique Bookmakers': plot_df_clean['BOOKMAKER'].nunique(),
                    'Unique Market Parameters': plot_df_clean['MARKETPARAMETERS'].nunique(),
                    'Date Range': f"{plot_df_clean['RECDATE_parsed'].min().strftime('%Y-%m-%d')} to {plot_df_clean['RECDATE_parsed'].max().strftime('%Y-%m-%d')}",
                    'Price Range': f"{plot_df_clean['PRICES_numeric'].min():.2f} - {plot_df_clean['PRICES_numeric'].max():.2f}"
                }
                
                for key, value in summary_data.items():
                    st.write(f"**{key}:** {value}")
                
        else:
            st.error("No valid numeric price data found after filtering. Please check your PRICES column format.")
            st.write("Sample PRICES values:", df['PRICES'].head(10).tolist())
    
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        
else:
    st.info("Please upload a CSV file to get started.")
    st.write("Expected columns: SPORT, SUPERODDSTYPE, MARKETPARAMETERS, MARKETPERIOD, INRUNNING, BOOKMAKER, RECDATE, PRICES, etc.")
