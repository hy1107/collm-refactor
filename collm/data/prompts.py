from collm.constants import USER_TOKEN, ITEM_TOKEN

_SYSTEM = (
    "You are a helpful movie recommendation assistant. "
    "Based on the user's watch history, predict whether they would enjoy the suggested movie."
)

_TEMPLATE_WITH_SIGNAL = (
    "{system}\n\n"
    "### Instruction:\n"
    "User {user_token} has watched: {history}.\n"
    "Would this user enjoy {item_token} {target_title}?\n\n"
    "### Response:\n"
)

_TEMPLATE_NO_SIGNAL = (
    "{system}\n\n"
    "### Instruction:\n"
    "A user has watched: {history}.\n"
    "Would this user enjoy {target_title}?\n\n"
    "### Response:\n"
)


def get_prompt(
    history_titles: list,
    target_title: str,
    use_collaborative_signal: bool,
    max_history: int = 20,
) -> str:
    history = ", ".join(history_titles[-max_history:]) if history_titles else "nothing"

    if use_collaborative_signal:
        return _TEMPLATE_WITH_SIGNAL.format(
            system=_SYSTEM,
            user_token=USER_TOKEN,
            history=history,
            item_token=ITEM_TOKEN,
            target_title=target_title,
        )
    else:
        return _TEMPLATE_NO_SIGNAL.format(
            system=_SYSTEM,
            history=history,
            target_title=target_title,
        )


def get_answer(label: int) -> str:
    return " Yes" if label == 1 else " No"
