"""
Gap Analysis: Hyperscaler Clean Energy Portfolios vs. Projected Demand

Loads deal-level CSV, aggregates by company/technology/status,
and compares against demand forecasts.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_deals():
    """Load and clean the deal-level dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'hyperscaler_ppa_deals.csv'))
    # Clean capacity — handle ranges and estimates
    df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
    return df


def load_forecasts():
    """Load demand forecast dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, 'demand_forecasts.csv'))


def is_framework_parent(row):
    """
    Identify framework-level entries that INCLUDE sub-deals listed separately.
    These should be excluded from simple sums to avoid double-counting.
    """
    framework_parents = [
        'X-energy SMR pipeline (framework target)',  # Includes Energy Northwest + Dominion
        'Kairos Power SMR fleet',  # Includes Kairos/TVA Hermes 2
        'ENGIE partnership (total)',  # Includes ENGIE Swenson Ranch
    ]
    return row['deal_name'] in framework_parents


def aggregate_by_company(df):
    """Aggregate capacity by company, excluding framework parents to avoid double-counting."""
    # Filter out framework parents
    df_clean = df[~df.apply(is_framework_parent, axis=1)]

    summary = df_clean.groupby('company').agg(
        total_capacity_mw=('capacity_mw', 'sum'),
        num_deals=('deal_name', 'count'),
        technologies=('technology', lambda x: ', '.join(sorted(x.dropna().unique()))),
    ).sort_values('total_capacity_mw', ascending=False)

    summary['total_capacity_gw'] = summary['total_capacity_mw'] / 1000
    return summary


def aggregate_by_status(df):
    """Aggregate capacity by company and status."""
    df_clean = df[~df.apply(is_framework_parent, axis=1)]

    pivot = df_clean.pivot_table(
        values='capacity_mw',
        index='company',
        columns='status',
        aggfunc='sum',
        fill_value=0
    )
    pivot = pivot / 1000  # Convert to GW
    pivot.columns = [f'{c}_gw' for c in pivot.columns]
    return pivot.sort_values(pivot.columns.tolist(), ascending=False)


def aggregate_by_technology(df):
    """Aggregate capacity by technology type across all companies."""
    df_clean = df[~df.apply(is_framework_parent, axis=1)]

    tech_summary = df_clean.groupby('technology').agg(
        total_capacity_mw=('capacity_mw', 'sum'),
        num_deals=('deal_name', 'count'),
        companies=('company', lambda x: ', '.join(sorted(x.unique()))),
    ).sort_values('total_capacity_mw', ascending=False)

    tech_summary['total_capacity_gw'] = tech_summary['total_capacity_mw'] / 1000
    return tech_summary


def us_iso_breakdown(df):
    """Aggregate US capacity by ISO market where identified."""
    us_deals = df[
        (df['country'].str.contains('US', na=False)) &
        (df['iso_market'].notna()) &
        (~df.apply(is_framework_parent, axis=1))
    ]

    iso_summary = us_deals.groupby('iso_market').agg(
        total_capacity_mw=('capacity_mw', 'sum'),
        num_deals=('deal_name', 'count'),
        companies=('company', lambda x: ', '.join(sorted(x.unique()))),
    ).sort_values('total_capacity_mw', ascending=False)

    iso_summary['total_capacity_gw'] = iso_summary['total_capacity_mw'] / 1000
    return iso_summary


def print_report(df, forecasts):
    """Print a formatted gap analysis report."""
    print("=" * 80)
    print("HYPERSCALER CLEAN ENERGY PORTFOLIO — GAP ANALYSIS")
    print("=" * 80)

    # Company summary
    print("\n1. CAPACITY BY COMPANY (de-duplicated)")
    print("-" * 60)
    company_summary = aggregate_by_company(df)
    for company, row in company_summary.iterrows():
        print(f"  {company:20s}  {row['total_capacity_gw']:8.1f} GW  ({row['num_deals']} deals)")
    total_gw = company_summary['total_capacity_gw'].sum()
    print(f"  {'TOTAL':20s}  {total_gw:8.1f} GW")

    # Status breakdown
    print("\n2. CAPACITY BY STATUS (GW)")
    print("-" * 60)
    status_summary = aggregate_by_status(df)
    print(status_summary.to_string())

    # Technology breakdown
    print("\n3. CAPACITY BY TECHNOLOGY")
    print("-" * 60)
    tech_summary = aggregate_by_technology(df)
    for tech, row in tech_summary.iterrows():
        print(f"  {tech:20s}  {row['total_capacity_gw']:8.1f} GW  ({row['num_deals']} deals)")

    # US ISO breakdown
    print("\n4. US CAPACITY BY ISO MARKET (where identified)")
    print("-" * 60)
    iso_summary = us_iso_breakdown(df)
    for iso, row in iso_summary.iterrows():
        print(f"  {iso:10s}  {row['total_capacity_gw']:8.1f} GW  ({row['companies']})")

    # Gap analysis
    print("\n5. DEMAND GAP ANALYSIS")
    print("-" * 60)
    print(f"  Total contracted capacity:  {total_gw:.0f} GW")
    print(f"  At ~25% avg CF:             {total_gw * 0.25 * 8760 / 1000:.0f} TWh/yr potential generation")
    print(f"  At ~35% avg CF:             {total_gw * 0.35 * 8760 / 1000:.0f} TWh/yr potential generation")
    print()
    print("  Demand forecasts (2030):")
    for _, row in forecasts[forecasts['year_2030'].notna()].iterrows():
        val = row['year_2030']
        print(f"    {row['source']:25s}  {row['scope']:15s}  {val} {row['unit']}")

    print()
    print("  KEY INSIGHT: Even at 35% CF, the ~145 GW contracted would generate")
    print("  ~444 TWh/yr — well short of the 1,000-1,600 TWh global demand projected")
    print("  for 2030. And much of this capacity is framework/planned, not built.")
    print()
    print("  CAVEAT: Not all contracted capacity serves data centers specifically.")
    print("  Company-level consumption data needed for true gap calculation.")


if __name__ == '__main__':
    deals = load_deals()
    forecasts = load_forecasts()
    print_report(deals, forecasts)
