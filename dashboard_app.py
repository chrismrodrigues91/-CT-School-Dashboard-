# =====================================
# 📚 IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import numpy as np   # 🔵 (Optional but useful if you need array operations)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import plotly.express as px  # ✅ NEW import

# 📂 LOAD THE DATA
df = pd.read_csv("All_Years_With_County_Area.csv", encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ✅ Fix spending column into numbers
df["Total_Per_Pupil_Expenditures"] = pd.to_numeric(df["Total_Per_Pupil_Expenditures"], errors="coerce")

# 🎨 SIDEBAR FILTERS
st.sidebar.title("🖌️ Filters")

# Year filter
year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()))

# County filter
counties = sorted(df["County"].dropna().unique())
selected_county = st.sidebar.selectbox("Select County", counties)

# Filter by year + county
county_df = df[(df["Year"] == year) & (df["County"] == selected_county)]

# District filter
unique_districts = county_df["District"].drop_duplicates().sort_values()
district_labels = unique_districts.tolist()
selected_labels = st.sidebar.multiselect(
    "Select District(s)",
    options=district_labels,
    default=district_labels if district_labels else [],
    help="Search or scroll to find districts"
)
districts = selected_labels

# Subject filter
subject = st.sidebar.radio("Select Subject", ["ELA_Scores", "Math_Scores"])

# Grade levels filter (if exists)
grade_levels = None
if "Grades" in df.columns:
    grade_levels = st.sidebar.multiselect(
        "Select Grade Level (%)",
        sorted(df["Grades"].dropna().unique()),
        default=df["Grades"].dropna().unique()
    )

# 📊 DASHBOARD TITLE
# 📊 Main Title
st.title("📊 CT Education Dashboard")
st.write(f"**County:** {selected_county}")

# 📝 Dashboard Description (revised and cleaner)
st.markdown("""
This dashboard provides an interactive view of student performance, spending, and demographics across Connecticut school districts.  
Use the filters on the left to explore patterns by **year**, **county**, **district(s)**, and **subject** (ELA or Math).  
""")



# import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'county_df' is your dataframe and 'subject' is the column you're working with

# Filtered data
filtered_df = county_df[county_df["District"].isin(districts)]
if grade_levels:
    filtered_df = filtered_df[filtered_df["Grades"].isin(grade_levels)]

# Highest and lowest scores
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'county_df' is your dataframe and 'subject' is the column you're working with

# Filtered data
filtered_df = county_df[county_df["District"].isin(districts)]
if grade_levels:
    filtered_df = filtered_df[filtered_df["Grades"].isin(grade_levels)]

# 1. Highest and lowest scores
# 1. Highest and lowest scores
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you already have the filtered data loaded in 'filtered_df'
highest = filtered_df.sort_values(by=subject, ascending=False).head(5)
lowest = filtered_df.sort_values(by=subject).head(5)

# Prepare data for plotting
highest_districts = highest['District']
highest_scores = highest[subject]

lowest_districts = lowest['District']
lowest_scores = lowest[subject]

# Plotting the bar chart for highest and lowest test scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(highest_districts, highest_scores, color='green', label='Top 5 Districts')
ax.bar(lowest_districts, lowest_scores, color='red', label='Bottom 5 Districts')

# 🔁 Dynamic labels based on selected subject
subject_label = subject.replace("_", " ").replace("Scores", "").strip()

# ✅ Dynamic title with county
ax.set_title(f"Top and Bottom 5 Districts by {subject_label} Scores in {selected_county} County")
ax.set_xlabel("Districts")
ax.set_ylabel(f"{subject_label} Scores")
ax.legend()

# Rotate x-axis labels
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# ✅ Dynamic Streamlit step title
st.subheader(f"1. Which Districts in {selected_county} County Have the Highest and Lowest SBAC Scores?")

# Show chart
st.pyplot(fig)
st.markdown(f"💡 **Insight:** There is a wide gap in performance across districts in **{selected_county} County**. While some districts score very high, others fall significantly behind.")



# 2. Per-Pupil Spending by District
# 2. Per-Pupil Spending by District
st.subheader("2. How Does Per-Pupil Spending Vary by District?")

fig1, ax1 = plt.subplots(figsize=(18, 12))  # Bigger for many districts

# Sort and fix spending column
sorted_df = filtered_df.sort_values(by="Total_Per_Pupil_Expenditures", ascending=True)
sorted_df["Total_Per_Pupil_Expenditures"] = pd.to_numeric(sorted_df["Total_Per_Pupil_Expenditures"], errors="coerce")

# Plot
sns.barplot(data=sorted_df, x="Total_Per_Pupil_Expenditures", y="District", ax=ax1, orient='h')

# Titles and labels
ax1.set_title(f"Total Per-Pupil Expenditures by District in {selected_county} County", fontsize=22)
ax1.set_xlabel("Total Per Pupil Expenditures ($)", fontsize=18)
ax1.set_ylabel("District", fontsize=18)

# X-axis settings
ax1.set_xlim(left=0)

# 📢 Show fewer x-axis ticks (only start maybe)
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

# No rotation needed
ax1.tick_params(axis='x', which='major', labelsize=14, rotation=0)
ax1.tick_params(axis='y', labelsize=12)

# Gridlines
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# ✅ Add spending labels at the end of each bar
for i, (value, district) in enumerate(zip(sorted_df["Total_Per_Pupil_Expenditures"], sorted_df["District"])): 
    if pd.notna(value):
        ax1.text(value + 500, i, f"${value:,.0f}", va='center', fontsize=10)

st.pyplot(fig1)
# ⭐ INSIGHT after Spending Bar Chart
# Show the chart (assuming it's already generated above)
# st.pyplot(fig)  # or st.plotly_chart(...)

## Add insight below based on the selected county
if selected_county == "New Haven":
    st.markdown("🔍 **Insight:** Spending varies widely in **New Haven County**, with Area Cooperative Educational Services spending nearly double compared to other districts.")
elif selected_county == "Fairfield":
    st.markdown("🔍 **Insight:** **Fairfield County** shows a clear divide—wealthier districts like Greenwich and New Canaan spend much more than others.")
elif selected_county == "Hartford":
    st.markdown("🔍 **Insight:** In **Hartford County**, there is a broader mid-range spending pattern, with fewer extreme outliers.")
elif selected_county == "Bridgeport":
    st.markdown("🔍 **Insight:** **Bridgeport County** districts have relatively close per-pupil spending values, showing less disparity than other counties.")
elif selected_county == "Litchfield":
    st.markdown("🔍 **Insight:** **Litchfield County** has a wide range, from under $20K to over $53K. EdAdvance and Gilbert School District are clear outliers.")
elif selected_county == "Middlesex":
    st.markdown("🔍 **Insight:** **Middlesex County** has fairly consistent mid-to-high spending, peaking around $31K.")
elif selected_county == "New London":
    st.markdown("🔍 **Insight:** **New London County** shows a narrower spread in spending, with most districts clustered between $18K and $24K.")
elif selected_county == "Tolland":
    st.markdown("🔍 **Insight:** **Tolland County** features a wide gap, from under $18K to nearly $49K, with EASTCONN as a major outlier.")
elif selected_county == "Windham":
    st.markdown("🔍 **Insight:** **Windham County** spending varies moderately, ranging from $18K to $29K, with Hampton being the highest.")
else:
    st.markdown(f"🔍 **Insight:** Per-pupil spending varies across districts in **{selected_county} County**, revealing funding differences that may impact outcomes.")




# 3. Correlation: Spending vs. Performance
# 3. Spending vs. Performance
st.subheader("3. Does Per-Pupil Spending Influence Student Test Performance?")

# Prepare the data
filtered_scatter = filtered_df.copy()
filtered_scatter = filtered_scatter.dropna(subset=["Total_Per_Pupil_Expenditures", subject])

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# Use a colormap that supports many categories
colors = plt.get_cmap('tab20')

# Create the plot
fig2, ax2 = plt.subplots(figsize=(14, 8))  # ✅ Bigger size

# Assign a unique label number to each district
unique_districts = sorted(filtered_scatter["District"].unique())
district_labels = {district: str(i+1) for i, district in enumerate(unique_districts)}

# Plot each district's label as a colored box
for i, (district, group) in enumerate(filtered_scatter.groupby("District")):
    color = colors(i % 20)
    for _, row in group.iterrows():
        ax2.text(
            row["Total_Per_Pupil_Expenditures"],
            row[subject],
            district_labels[district],
            fontsize=9,
            ha='center',
            va='center',
            weight='bold',
            color='black',
            bbox=dict(
                facecolor=color,
                edgecolor='black',
                boxstyle='round,pad=0.3',
                linewidth=0.8
            )
        )

# Regression line over all data
sns.regplot(
    data=filtered_scatter,
    x="Total_Per_Pupil_Expenditures",
    y=subject,
    scatter=False,
    color="black",
    line_kws={"linewidth": 2},
    ci=95,
    ax=ax2
)

# ✅ Dynamic Y-axis scaling so no points are cut off
max_y = filtered_scatter[subject].max()
ax2.set_ylim(0, max_y * 1.1)

# 🔁 Dynamic title using selected_county
ax2.set_title(f"Relationship Between Per-Pupil Spending and District Test Scores in {selected_county} County", fontsize=20)

# Labels
ax2.set_xlabel("Total Per Pupil Expenditures ($)", fontsize=14)
ax2.set_ylabel(subject.replace("_", " "), fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# Custom legend showing district names with colors
legend_patches = [
    mpatches.Patch(color=colors(i % 20), label=f"{i+1}. {d}")
    for i, d in enumerate(unique_districts)
]

ax2.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=10,
    title="District",
    title_fontsize=12,
    ncol=1,
    frameon=True
)

# Show final chart in Streamlit
st.pyplot(fig2)
st.markdown("""
**Legend Continued**  
- **Black Line** – Regression Line (shows the overall trend in the data)  
- **Shaded Gray Area** – 95% Confidence Interval (range where we expect the true trend to fall)  
""")
# 💡 INSIGHT for Step 3 – Scatter Plot (Spending vs. Performance)
if selected_county == "New Haven":
    st.markdown("💡 **Insight:** In **New Haven County**, the trend line is nearly flat, indicating **little to no correlation** between spending and test scores. Some districts spend a lot but perform poorly—**Area Cooperative Educational Services** is a clear outlier—while others perform well with average spending. This suggests that **factors beyond funding**, such as instruction quality or student support, might play a larger role.")
elif selected_county == "Fairfield":
    st.markdown("💡 **Insight:** In **Fairfield County**, there’s a slight positive trend, but some high-spending districts still score below average, suggesting spending is not the only factor.")
elif selected_county == "Litchfield":
    st.markdown("💡 **Insight:** In **Litchfield County**, the trend is clearly negative—districts with higher spending actually tend to score lower, making this county an outlier.")
elif selected_county == "Middlesex":
    st.markdown("💡 **Insight:** In **Middlesex County**, the correlation is negative, with districts spending more generally scoring lower, suggesting other factors are influencing outcomes.")
elif selected_county == "New London":
    st.markdown("💡 **Insight:** **New London County** shows a mild positive trend, but the scores remain low overall, even among higher-spending districts.")
elif selected_county == "Tolland":
    st.markdown("💡 **Insight:** **Tolland County** exhibits a downward trend—districts that spend more tend to have lower performance, showing an inverse relationship.")
elif selected_county == "Windham":
    st.markdown("💡 **Insight:** In **Windham County**, the relationship is slightly negative, with many districts spending over $20K but still scoring low.")
else:
    st.markdown(f"💡 **Insight:** In **{selected_county} County**, the relationship between spending and performance is not strongly positive, suggesting spending alone may not drive achievement.")


#4.
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# STEP 4 TITLE
st.subheader("4. Demographics and Their Impact on Student Performance")
st.markdown(f"""
This step explores how student racial/ethnic composition in a district in **{selected_county} County** relates to academic performance.  
The first chart shows how the proportion of each demographic group correlates with test scores.  
The second chart summarizes which majority-demographic districts tend to perform better on average.
""")

# ✅ Define consistent colors for both charts
line_colors = {
    "Black": "blue",
    "Hispanic": "orange",
    "White": "green",
    "Asian": "purple",
    "Other": "red"
}

# ✅ Melt the dataframe for proportions
bubble_df = filtered_df.copy()
bubble_melted = bubble_df.melt(
    id_vars=["District", "County", subject],
    value_vars=["Black", "Hispanic", "White", "Asian", "Other"],
    var_name="Demographic",
    value_name="Proportion"
)

# ✅ Clean and convert proportions
bubble_melted["Proportion (%)"] = pd.to_numeric(bubble_melted["Proportion"] * 100, errors='coerce')
bubble_melted_clean = bubble_melted.dropna(subset=["Proportion (%)", subject])

# ✅ Chart 1: Bubble chart with matching trend lines
fig1 = px.scatter(
    bubble_melted_clean,
    x="Proportion (%)",
    y=subject,
    color="Demographic",
    color_discrete_map=line_colors,
    size="Proportion (%)",
    hover_name="District",
    hover_data={
        "County": True,
        "Demographic": True,
        subject: True,
        "Proportion (%)": True
    },
    title=f"How Race and Ethnicity Relate to Test Performance in {selected_county} County",
    width=950,
    height=600,
)

# Trend lines per group
for group in line_colors:
    group_df = bubble_melted_clean[bubble_melted_clean["Demographic"] == group]
    x = group_df["Proportion (%)"]
    y = group_df[subject]
    if len(x) > 1 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        fig1.add_trace(
            go.Scatter(
                x=x,
                y=regression_line,
                mode="lines",
                name=f"Trend Line ({group})",
                line=dict(color=line_colors[group], dash="dash")
            )
        )

fig1.update_layout(
    title_font_size=18,
    legend_title="Demographic",
    margin=dict(l=40, r=40, t=60, b=40)
)

# ✅ Show Chart 1
st.plotly_chart(fig1, use_container_width=True)

# ✅ Add custom legend explanation below the chart
st.markdown(f"""
📌 **Legend**  
- **Dashed Lines** – Regression Lines (show the trend for each demographic group)  
- **Colored Bubbles** – Each district in **{selected_county} County**, with size based on demographic proportion  
- **X-axis** – Percentage of students from the demographic group  
- **Y-axis** – {subject.replace("_", " ")}  
""")
if selected_county == "Bridgeport":
    st.markdown("💡 **Insight:** In **Bridgeport County**, White students show a slight positive relationship between proportion and performance, while Hispanic students show a clear negative trend—higher proportions are associated with lower scores.")
elif selected_county == "Fairfield":
    st.markdown("💡 **Insight:** **Fairfield County** shows a negative trend for most groups, especially Black and White students, indicating that higher representation of these demographics often coincides with lower test performance.")
elif selected_county == "Hartford":
    st.markdown("💡 **Insight:** In **Hartford County**, the trend is mixed—Hispanic students show a strong positive association, while Black and White groups show a weak or negative relationship.")
elif selected_county == "Litchfield":
    st.markdown("💡 **Insight:** **Litchfield County** reveals strong negative trends for White students, while Black and Hispanic students show positive slopes, though based on fewer data points.")
elif selected_county == "Middlesex":
    st.markdown("💡 **Insight:** In **Middlesex County**, the White student group shows a steep negative relationship with scores, while Black and Asian groups show a sharp positive trend.")
elif selected_county == "New London":
    st.markdown("💡 **Insight:** **New London County** exhibits a mixed pattern—Hispanic students trend upward while White students trend downward, suggesting differing equity gaps.")
elif selected_county == "Tolland":
    st.markdown("💡 **Insight:** In **Tolland County**, most demographic groups show a positive trend, suggesting that greater representation aligns with better performance.")
elif selected_county == "Windham":
    st.markdown("💡 **Insight:** **Windham County** displays a very strong negative trend for White students, while Hispanic and Asian students show clear positive relationships.")
else:
    st.markdown(f"💡 **Insight:** In **{selected_county} County**, demographic trends vary. Some groups show that higher representation leads to stronger performance, while others reveal potential equity gaps.")

# ✅ Chart 2: Average Scores by Majority Group
st.markdown("---")  # horizontal line for visual separation

# ✅ Static section heading
st.markdown("### Distribution of ELA Scores by Majority Demographic Group")

# Copy the filtered data
majority_demo_df = filtered_df.copy()

# Define demographic columns
demo_cols = ["Black", "Hispanic", "White", "Asian", "Other"]

# Ensure columns are numeric
for col in demo_cols:
    majority_demo_df[col] = pd.to_numeric(majority_demo_df[col], errors='coerce')

# Get the majority demographic for each district
majority_demo_df["Majority Group"] = majority_demo_df[demo_cols].idxmax(axis=1)

# Calculate average ELA/Math scores per majority group
avg_scores = (
    majority_demo_df
    .groupby("Majority Group")[subject]
    .mean()
    .reset_index()
    .sort_values(subject, ascending=False)
)

# ✅ Chart title inside chart
chart_title = f"Average {subject.replace('_', ' ')} Scores by Majority Demographic Group in {selected_county} County"

# Plot bar chart
fig2 = px.bar(
    avg_scores,
    x="Majority Group",
    y=subject,
    color="Majority Group",
    color_discrete_map=line_colors,
    labels={subject: "Average Score"},
    title=chart_title,
    height=500
)

# Display in Streamlit
st.plotly_chart(fig2, use_container_width=True)


# 5. Demographics: High vs. Low Performing
# Step 5: Demographics in High vs Low Performing Districts
import matplotlib.pyplot as plt
import streamlit as st

# Subheader for Step 5
st.subheader(f"5. What Do Student Demographics Look Like in High- vs. Low-Performing Districts in {selected_county} County?")

# Demographic categories and colors
demographic_cols = ["Black", "Hispanic", "White", "Asian", "Other"]
colors = ["#66b3ff", "#ffb3e6", "#ffcc99", "#99ff99", "#ff6666"]

# Filter top 5 and bottom 5 districts by selected subject
highest = filtered_df.sort_values(by=subject, ascending=False).head(5)
lowest = filtered_df.sort_values(by=subject).head(5)

# Convert columns to numeric
highest[demographic_cols] = highest[demographic_cols].apply(pd.to_numeric, errors='coerce')
lowest[demographic_cols] = lowest[demographic_cols].apply(pd.to_numeric, errors='coerce')

# Calculate average demographic percentages
high_perf = (highest[demographic_cols].mean() / highest[demographic_cols].mean().sum() * 100).round(1).tolist()
low_perf = (lowest[demographic_cols].mean() / lowest[demographic_cols].mean().sum() * 100).round(1).tolist()

# Explode slices for visibility (especially small ones)
explode = (0.05, 0.05, 0.2, 0.25, 0.3)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# High Performing Pie Chart
wedges1, texts1, autotexts1 = ax1.pie(
    high_perf,
    labels=None,
    colors=colors,
    startangle=90,
    wedgeprops={'width': 0.4},
    autopct='%1.1f%%',
    pctdistance=1.25,
    explode=explode
)
ax1.set_title("Student Demographics in Top-Scoring Districts", y=1.18)

# Low Performing Pie Chart
wedges2, texts2, autotexts2 = ax2.pie(
    low_perf,
    labels=None,
    colors=colors,
    startangle=90,
    wedgeprops={'width': 0.4},
    autopct='%1.1f%%',
    pctdistance=1.25,
    explode=explode
)
ax2.set_title("Student Demographics in Low-Scoring Districts", y=1.18)

# Style percentage labels
for text in autotexts1 + autotexts2:
    text.set_fontsize(10)
    text.set_weight('bold')

# Add shared legend at the bottom
fig.legend(
    demographic_cols,
    title="Demographic",
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.05),
    frameon=False
)

# plt.tight_layout()  # disabled to avoid memory allocation errors

st.pyplot(fig)
if selected_county == "New Haven":
    st.markdown("💡 **Insight:** In **New Haven County**, top districts have more Hispanic and White students, while low-performing ones have higher proportions of Black and White students. This indicates a potential disparity in academic outcomes tied to district demographics.")
elif selected_county == "Bridgeport":
    st.markdown("💡 **Insight:** In **Bridgeport County**, both high- and low-performing districts have nearly identical demographics—predominantly Black and Hispanic students—suggesting performance differences stem from non-demographic factors.")
elif selected_county == "Fairfield":
    st.markdown("💡 **Insight:** In **Fairfield County**, high-performing districts have more Hispanic students, while low-performing ones have more Black and Asian students. This may reflect equity challenges across communities.")
elif selected_county == "Hartford":
    st.markdown("💡 **Insight:** In **Hartford County**, top-scoring districts are more diverse, with a higher percentage of Hispanic students, while low-performers are majority White. This suggests diverse schools in this region are performing better.")
elif selected_county == "Litchfield":
    st.markdown("💡 **Insight:** In **Litchfield County**, both high and low-performing districts are majority White, but top districts show slightly more diversity, implying that demographic variety might align with higher outcomes.")
elif selected_county == "Middlesex":
    st.markdown("💡 **Insight:** In **Middlesex County**, high-performing districts are slightly more diverse, while low-performing ones are overwhelmingly White, hinting that diversity may correlate with stronger academic results.")
elif selected_county == "New London":
    st.markdown("💡 **Insight:** In **New London County**, high performers show greater balance across racial groups, while low-performing districts are majority White. More inclusive districts appear to perform better.")
elif selected_county == "Tolland":
    st.markdown("💡 **Insight:** In **Tolland County**, low-performing districts have a stronger White majority, while top-scoring ones show more demographic mix, especially among Asian and Hispanic students.")
elif selected_county == "Windham":
    st.markdown("💡 **Insight:** In **Windham County**, top-performing districts have more Hispanic and Black students, while the lowest performers are heavily White, revealing a sharp demographic divide in academic performance.")
else:
    st.markdown(f"💡 **Insight:** In **{selected_county} County**, student performance may relate to demographic balance. High-performing districts often show more diversity, while low performers tend to have less.")


# 6. Trends over Time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load and clean data
df = pd.read_csv("All_Years_With_County_Area.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
df["County"] = df["County"].str.strip()

# Sidebar filter assumed already declared earlier:
# selected_county = st.sidebar.selectbox("Select County", ...)

st.subheader("6. School Performance Over Time")

# All CT districts trend
district_avg_by_year = df.groupby(["Year", "District"])["ELA_Scores"].mean().reset_index()
all_ct_trend = district_avg_by_year.groupby("Year")["ELA_Scores"].mean()

# Selected county districts trend
county_districts = df[df["County"] == selected_county]
county_district_avg_by_year = (
    county_districts.groupby(["Year", "District"])["ELA_Scores"]
    .mean()
    .reset_index()
    .groupby("Year")["ELA_Scores"]
    .mean()
)

# Target range band
target_min = county_district_avg_by_year.mean() - 3
target_max = county_district_avg_by_year.mean() + 3

# === PLOT ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"School Performance of All CT Districts vs Districts in {selected_county} County", fontsize=14)


# Target range band
ax.axhspan(target_min, target_max, color='lightgreen', alpha=0.3, label="Target Range")

# Plot lines
ax.plot(all_ct_trend.index, all_ct_trend.values, label="All CT Districts", color='blue', linewidth=2)
ax.plot(county_district_avg_by_year.index, county_district_avg_by_year.values,
        label=f"Districts in {selected_county} County", color='orange', linewidth=2)

# Inline labels (adjusted to not float outside)
for year, val in all_ct_trend.items():
    ax.text(year, val + 2, f"{val:.0f}", color='blue', ha='center', fontsize=9)

for year, val in county_district_avg_by_year.items():
    ax.text(year, val + 4, f"{val:.0f}", color='orange', ha='center', fontsize=9)

# Axis & legend
ax.set_xlabel("Year")
ax.set_ylabel("ELA Scores")
ax.legend()
ax.grid(True)

# Display chart in Streamlit
st.pyplot(fig)
if selected_county == "Fairfield":
    st.markdown("💡 **Insight:** **Fairfield County** maintains exceptionally high test scores, over 1,000 points above the state average, with only a slight drop in the latest year.")
elif selected_county == "Hartford":
    st.markdown("💡 **Insight:** In **Hartford County**, scores remained stable for two years before slightly dropping, but still consistently perform above the statewide average.")
elif selected_county == "Litchfield":
    st.markdown("💡 **Insight:** **Litchfield County** has the lowest performance in the state across all three years, though it shows a slight improvement in the most recent year.")
elif selected_county == "Middlesex":
    st.markdown("💡 **Insight:** In **Middlesex County**, scores remained flat for two years and then dipped slightly, continuing to lag well behind the state average.")
elif selected_county == "New Haven":
    st.markdown("💡 **Insight:** **New Haven County** shows a gradual decline in scores, mirroring the state trend but maintaining a consistent gap below the statewide average.")
elif selected_county == "New London":
    st.markdown("💡 **Insight:** In **New London County**, performance has steadily declined over the three years, with scores remaining far below the Connecticut average.")
elif selected_county == "Tolland":
    st.markdown("💡 **Insight:** **Tolland County** shows a noticeable drop in 2023-24 after two years of stable performance, and it consistently trails the state average by a large margin.")
elif selected_county == "Windham":
    st.markdown("💡 **Insight:** **Windham County** shows a minor improvement between 2021-22 and 2022-23 but then flattens, with performance levels remaining the lowest in the state.")
else:
    st.markdown(f"💡 **Insight:** In **{selected_county} County**, school performance over time reveals whether districts are improving, declining, or holding steady compared to the state average.")




# 7. Urban vs Suburban Comparison
st.subheader("7. Urban vs. Suburban Comparison")

if "Area_Type" in df.columns:
    # Filter and group the data
    area_df = df[(df["Year"] == year) & df["Area_Type"].notna()].groupby("Area_Type")[[subject]].mean().reset_index()

    if not area_df.empty and len(area_df) > 1:
        # BAR CHART
        st.markdown("**Bar Chart: Average Scores by Area Type**")
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(data=area_df, x="Area_Type", y=subject, ax=ax_bar)
        ax_bar.set_title("Urban vs. Suburban Performance")
        ax_bar.set_ylabel(subject.replace("_", " "))
        st.pyplot(fig_bar)
        st.markdown("💡 **Insight:** Surprisingly, **urban districts** outperform **suburban districts** in average test scores. This challenges common assumptions and may reflect targeted resources, programs, or testing strategies in urban areas that are driving higher performance.")



















      
















