
# Fan out across all 8 coin DataFrames using spread()
# spread() returns a SlicedIterable which acts as the value in downstream blocks.
# The variable IS the iterable — downstream it resolves to the actual element.
# We must NOT try to use the spread result as a dict key or subscript in the SAME block.
# Instead, only call spread() and export the result; downstream blocks get the resolved value.

coin_symbol = spread(["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "NEIRO", "ZEREBRO"])

print(f"[Fleet slice] Spread initiated for coin_symbol over 8 coins")
