import logging
from pathlib import Path
from typing import List, Optional
from sidekick.prompter import db_setup_api, query_api

import openai
import toml
from h2o_wave import Q, app, data, handle_on, main, on, ui

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/ui/.app_config.toml")
ui_title = env_settings["WAVE_UI"]["TITLE"]
ui_description = env_settings["WAVE_UI"]["SUB_TITLE"]

db_settings = toml.load(f"{base_path}/sidekick/configs/.env.toml")
db_dialect = db_settings["DB-DIALECT"]["DB_TYPE"]
host_name = db_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
user_name = db_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
password = db_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
db_name = db_settings["LOCAL_DB_CONFIG"]["DB_NAME"]
port = db_settings["LOCAL_DB_CONFIG"]["PORT"]
table_info_path = f'{base_path}/{db_settings["TABLE_INFO"]["TABLE_INFO_PATH"]}'
table_samples_path = f'{base_path}/{db_settings["TABLE_INFO"]["TABLE_SAMPLES_PATH"]}'
table_name = db_settings["TABLE_INFO"]["TABLE_NAME"]

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")


# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...
def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return

    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


@on("#chat")
async def chat(q: Q):
    q.page["sidebar"].value = "#chat"
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, "background_card", ui.form_card(box="horizontal", items=[ui.text("Ask your questions:")]))
    add_card(
        q,
        "chat_card",
        ui.chatbot_card(
            box=ui.box("vertical", height="500px"),
            name="chatbot",
            data=data(fields="content from_user", t="list", size=-50),
        ),
    )


@on("chatbot")
async def chatbot(q: Q):
    q.page["sidebar"].value = "#chat"
    # clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    # Append user message.
    q.page["chat_card"].data += [q.args.chatbot, True]
    # Append bot response.
    question = f"{q.args.chatbot}"
    logging.info(f"Question: {question}")

    if q.args.chatbot.lower() == "db setup":
        llm_response = db_setup_api(db_name=db_name, hostname=host_name, user_name=user_name, password=password, port=port, table_info_path=table_info_path, table_samples_path=table_samples_path, table_name= table_name)
    else:
        llm_response = query_api(question = question,
                                sample_queries=None,
                                table_info_path=table_info_path)
        llm_response = "\n".join(llm_response)

    q.page["chat_card"].data += [llm_response, False]


@on("#datasets")
async def datasets(q: Q):
    q.page["sidebar"].value = "#datasets"
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    add_card(q, "data_header", ui.form_card(box="horizontal", title="Dataset", items=[]))

    add_card(
        q,
        "table",
        ui.form_card(
            box="vertical",
            items=[
                ui.table(
                    name="table",
                    columns=[
                        ui.table_column(
                            name=i,
                            label=i,
                            sortable=True,
                            data_type="string" if q.client.data[i].dtype == "O" else "number",
                        )
                        for i in q.client.data.columns
                    ],
                    rows=[
                        ui.table_row(name="row{}".format(idx), cells=[str(i) for i in row])
                        for idx, row in q.client.data.iterrows()
                    ],
                )
            ],
        ),
    )


@on("#about")
async def about(q: Q):
    q.page["sidebar"].value = "#about"
    clear_cards(q)


@on("#support")
async def handle_page4(q: Q):
    q.page["sidebar"].value = "#support"
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    # Since this page is interactive, we want to update its card instead of recreating it every time, so ignore 'form' card on drop.
    clear_cards(q, ["form"])


async def init(q: Q) -> None:
    q.page["meta"] = ui.meta_card(
        box="",
        layouts=[
            ui.layout(
                breakpoint="xs",
                min_height="100vh",
                zones=[
                    ui.zone(
                        "main",
                        size="1",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone("sidebar", size="250px"),
                            ui.zone(
                                "body",
                                zones=[
                                    ui.zone(
                                        "content",
                                        zones=[
                                            # Specify various zones and use the one that is currently needed. Empty zones are ignored.
                                            ui.zone("horizontal", direction=ui.ZoneDirection.ROW),
                                            ui.zone("vertical"),
                                            ui.zone(
                                                "grid", direction=ui.ZoneDirection.ROW, wrap="stretch", justify="center"
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    q.page["sidebar"] = ui.nav_card(
        box="sidebar",
        color="primary",
        title="QnA Assistant",
        subtitle="Get answers to your questions.",
        value=f'#{q.args["#"]}' if q.args["#"] else "#chat",
        image="https://wave.h2o.ai/img/h2o-logo.svg",
        items=[
            ui.nav_group(
                "Menu",
                items=[
                    ui.nav_item(name="#chat", label="Chat"),
                    ui.nav_item(name="#datasets", label="Dataset Snapshot"),
                ],
            ),
            ui.nav_group(
                "Help",
                items=[
                    ui.nav_item(name="#about", label="About"),
                    ui.nav_item(name="#support", label="Support"),
                ],
            ),
        ],
        secondary_items=[
            ui.persona(
                title="John Doe",
                subtitle="Developer",
                size="s",
                image="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&h=750&w=1260",
            ),
        ],
    )

    # Connect to LLM
    openai.api_key = ""

    ######################## COMMENTED OUT FOR NOW ###################
    # q.client.data = get_data()
    # q.client.mapping = get_mapping_dicts(q.client.data)
    # q.client.masked_data =

    # If no active hash present, render chat.
    if q.args["#"] is None:
        await chat(q)


@app("/")
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
    await q.page.save()
