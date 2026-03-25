# collm/constants.py
# USER_TOKEN 和 ITEM_TOKEN 定義在此處，供 data/ 和 model/ 層共用，
# 避免 data/prompts.py 直接 import model/backbone.py（跨層依賴）。

USER_TOKEN = "[USER_TOKEN]"
ITEM_TOKEN = "[ITEM_TOKEN]"
