import polars as pl
import streamlit as st
from shared import PROMOS, current_client, document, findings, hero

client = current_client()
f = findings(client)
summary = document(client, "summary.json")
years = f"{summary['first_week'][:4]} to {summary['last_week'][:4]}"

st.html(
    f'<h1 class="page-title">Cereal pricing at {f["short_name"]}</h1>'
    f'<p class="lede">This tool studied {summary["weeks"]} weeks of cereal sales in '
    f"{f['stores']} stores ({years}) to learn how shoppers react to prices and deals. "
    "Below is what it found, in plain terms.</p>"
)

cautious = f["band"] > abs(f["gain"])
hero(
    "The short answer",
    f"Raise {f['raised']} of the {f['products']} cereal prices by a few cents",
    f"That should add about ${f['gain']:,.0f} of profit a week across all stores "
    f"({f['gain_pct']:+.0%}), while selling about {abs(f['units_change_pct']):.0%} fewer boxes. "
    + (
        "It's an estimate, so try the new prices in a few stores before changing them everywhere."
        if cautious
        else ""
    ),
)

mechanics = f["mechanics"].with_columns(pl.col("scenario").replace(PROMOS))
paying = mechanics.filter(pl.col("margin") > 0)
cuts = mechanics.filter(pl.col("scenario").is_in(["10% off", "20% off"]))
best_deal = paying["scenario"][0] if paying.height else None
cuts_lose = cuts.height and cuts["margin"].max() < 0
promo_answer = (
    f"Only some. {best_deal} pays best, about ${paying['margin'][0]:,.0f} a week."
    if best_deal
    else "Not on their own. Every kind of deal loses profit on average."
) + (" Plain price cuts lose money." if best_deal and cuts_lose else "")

st.subheader("Three questions this tool answers")
cards = st.columns(3, gap="medium")
for col, question, answer, page, link in [
    (
        cards[0],
        "Should any prices change?",
        f"Yes. {f['raised']} prices go up a little and {f['held'] + f['cut']} stay about the same.",
        "views/prices.py",
        "See the new price list",
    ),
    (
        cards[1],
        "Do promotions make money?",
        promo_answer,
        "views/promos.py",
        "See which deals pay",
    ),
    (
        cards[2],
        "What if I change a price myself?",
        "Pick any cereal, move its price and see how many boxes you'd sell and where the "
        "shoppers go.",
        "views/try_price.py",
        "Try a price",
    ),
]:
    with col.container(border=True, height="stretch"):
        st.html(f'<div class="question">{question}</div><div class="answer">{answer}</div>')
        st.page_link(page, label=link, icon=":material/arrow_forward:")

st.subheader("How it works")
steps = st.columns(3, gap="medium")
for col, number, title, text in [
    (
        steps[0],
        1,
        "Learn from past sales",
        "For every cereal and week, it looks at the price, any deal running and how many boxes "
        "sold. From that it learns how much sales drop when a price goes up.",
    ),
    (
        steps[1],
        2,
        "Simulate shoppers",
        "When one cereal gets dearer, some shoppers buy a different cereal and some skip cereal "
        "that week. The tool estimates both, so it doesn't count the same shopper twice.",
    ),
    (
        steps[2],
        3,
        "Search for better prices",
        "It tries thousands of price combinations, keeps to sensible store rules (no price "
        "moves more than 10%, prices end in 9) and picks the one with the most profit.",
    ),
]:
    with col:
        st.html(f'<div class="card-title"><span class="step">{number}</span>{title}</div>')
        st.caption(text)

with st.container(border=True):
    st.markdown("**How sure are we?**")
    st.markdown(
        f"When tested on a year of sales the tool hadn't seen, its weekly forecasts were off by "
        f"about {f['wape_m3']:.0%} on average. In money terms the profit estimate can swing by "
        f"about \\${f['band']:,.0f} a week, "
        + (
            f"which is bigger than the \\${f['gain']:,.0f} gain. So treat the new prices as a "
            "good bet to test, not a sure thing."
            if cautious
            else f"which is smaller than the \\${f['gain']:,.0f} gain."
        )
    )
st.caption(
    "Profit here means sales minus what the store paid for the cereal, before staff and rent. "
    "For dunnhumby the store's cost isn't in the data, so it's assumed."
)
