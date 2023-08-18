import json
import logging
from pathlib import Path
from typing import List, Optional

import openai
import toml
from h2o_wave import Q, app, data, handle_on, main, on, ui
from sidekick.prompter import db_setup_api, query_api
from sidekick.utils import setup_dir, update_tables, get_table_keys

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/ui/app_config.toml")
tmp_path = f"{base_path}/var/lib/tmp"

ui_title = env_settings["WAVE_UI"]["TITLE"]
ui_description = env_settings["WAVE_UI"]["SUB_TITLE"]


async def user_variable(q: Q):
    db_settings = toml.load(f"{base_path}/sidekick/configs/env.toml")

    q.user.db_dialect = db_settings["DB-DIALECT"]["DB_TYPE"]
    q.user.host_name = db_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
    q.user.user_name = db_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
    q.user.password = db_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
    q.user.db_name = db_settings["LOCAL_DB_CONFIG"]["DB_NAME"]
    q.user.port = db_settings["LOCAL_DB_CONFIG"]["PORT"]

    tables, tables_info = get_table_keys(f"{tmp_path}/data/tables.json", None)
    table_info = tables_info[tables[0]] if len(tables) > 0 else None

    q.user.table_info_path = table_info["schema_info_path"] if len(tables) > 0 else None
    q.user.table_samples_path = table_info["samples_path"] if len(tables) > 0 else None
    q.user.sample_qna_path = table_info["samples_qa"] if len(tables) > 0 else None
    q.user.table_name = tables[0] if len(tables) > 0 else None

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")


async def client_variable(q: Q):
    q.client.query = None


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

    table_names = []
    tables, _ = get_table_keys(f"{tmp_path}/data/tables.json", None)
    for table in tables:
        table_names.append(ui.choice(table, f"Table: {table}"))

    add_card(q, "background_card", ui.form_card(box="horizontal", items=[ui.text("Ask your questions:")]))

    add_card(
        q,
        "select_tables",
        ui.form_card(
            box="vertical",
            items=[
                ui.dropdown(
                    name="table_dropdown",
                    label="Table",
                    required=True,
                    choices=table_names,
                    value=q.user.table_name if q.user.table_name else None,
                ),
                ui.button(name="submit_table", label="Submit", primary=True),
            ],
        ),
    )
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

    if q.page["select_tables"].table_dropdown.value is None or q.user.table_name is None:
        q.page["chat_card"].data += ["Please select a table to continue!", False]
        return

    # Append bot response.
    question = f"{q.args.chatbot}"
    logging.info(f"Question: {question}")

    if q.args.chatbot.lower() == "db setup":
        llm_response, err = db_setup_api(
            db_name=q.user.db_name,
            hostname=q.user.host_name,
            user_name=q.user.user_name,
            password=q.user.password,
            port=q.user.port,
            table_info_path=q.user.table_info_path,
            table_samples_path=q.user.table_samples_path,
            table_name=q.user.table_name,
        )
    elif q.args.chatbot.lower() == "regenerate":
        if q.client.query is not None and q.client.query.strip() != "":
            llm_response, err = query_api(
                question=q.client.query,
                sample_queries_path=q.user.sample_qna_path,
                table_info_path=q.user.table_info_path,
                table_name=q.user.table_name,
                is_regenerate=True,
            )
            llm_response = "\n".join(llm_response)
        else:
            llm_response, err = ("Sure, I can generate a new response for you. However, in order to assist you "
                                 "effectively could you please provide me with your question?"), None
    else:
        q.client.query = question
        llm_response, err = query_api(
            question=question,
            sample_queries_path=q.user.sample_qna_path,
            table_info_path=q.user.table_info_path,
            table_name=q.user.table_name,
        )
        llm_response = "\n".join(llm_response)

    q.page["chat_card"].data += [llm_response, False]


@on("file_upload")
async def fileupload(q: Q):
    q.page["dataset"].error_bar.visible = False
    q.page["dataset"].success_bar.visible = False
    q.page["dataset"].progress_bar.visible = True

    await q.page.save()

    q.page["sidebar"].value = "#datasets"
    usr_info_path = None
    usr_samples_path = None
    usr_sample_qa = None

    sample_data = q.args.sample_data
    sample_schema = q.args.data_schema
    sample_qa = q.args.sample_qa

    usr_table_name = q.args.table_name

    if sample_data is None or sample_schema is None or usr_table_name is None or usr_table_name.strip() == "":
        q.page["dataset"].error_bar.visible = True
        q.page["dataset"].progress_bar.visible = False
    else:
        usr_table_name = usr_table_name.lower()
        if sample_data:
            usr_samples_path = await q.site.download(
                sample_data[0], f"{tmp_path}/jobs/{usr_table_name}_table_samples.csv"
            )
        if sample_schema:
            usr_info_path = await q.site.download(
                sample_schema[0], f"{tmp_path}/jobs/{usr_table_name}_table_info.jsonl"
            )
        if sample_qa:
            usr_sample_qa = await q.site.download(sample_qa[0], f"{tmp_path}/jobs/{usr_table_name}_sample_qa.csv")

        q.page["dataset"].error_bar.visible = False

        table_metadata = dict()
        table_metadata[usr_table_name] = {
            "schema_info_path": usr_info_path,
            "samples_path": usr_samples_path,
            "samples_qa": usr_sample_qa,
        }
        update_tables(f"{tmp_path}/data/tables.json", table_metadata)

        q.user.table_name = usr_table_name
        q.user.table_samples_path = usr_samples_path
        q.user.table_info_path = usr_info_path
        q.user.sample_qna_path = usr_sample_qa

        db_resp = db_setup_api(
            db_name=q.user.db_name,
            hostname=q.user.host_name,
            user_name=q.user.user_name,
            password=q.user.password,
            port=q.user.port,
            table_info_path=q.user.table_info_path,
            table_samples_path=q.user.table_samples_path,
            table_name=q.user.table_name,
        )
        logging.info(f"DB updates: \n {db_resp}")
        q.page["dataset"].progress_bar.visible = False
        q.page["dataset"].success_bar.visible = True


@on("#datasets")
async def datasets(q: Q):
    q.page["sidebar"].value = "#datasets"
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    add_card(q, "data_header", ui.form_card(box="horizontal", title="Dataset", items=[]))

    add_card(
        q,
        "dataset",
        ui.form_card(
            box="vertical",
            items=[
                ui.message_bar(
                    name="error_bar",
                    type="error",
                    text="Please input table name, data & schema files to upload!",
                    visible=False,
                ),
                ui.message_bar(name="success_bar", type="success", text="Files Uploaded Successfully!", visible=False),
                ui.textbox(name="table_name", label="Table Name", required=True),
                ui.file_upload(
                    name="data_schema",
                    label="Data Schema",
                    multiple=False,
                    compact=True,
                    file_extensions=["jsonl"],
                    required=True,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!",
                ),
                ui.file_upload(
                    name="sample_qa",
                    label="Sample Q&A",
                    multiple=False,
                    compact=True,
                    file_extensions=["csv"],
                    required=False,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!",
                ),
                ui.file_upload(
                    name="sample_data",
                    label="Data Samples",
                    multiple=False,
                    compact=True,
                    file_extensions=["csv"],
                    required=True,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!",
                ),
                ui.progress(
                    name="progress_bar", width="100%", label="Uploading datasets and creating tables!", visible=False
                ),
                ui.button(name="file_upload", label="Submit", primary=True),
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


@on("submit_table")
async def submit_table(q: Q):
    table_key = q.args.table_dropdown
    if table_key:
        _, table_info = get_table_keys(f"{tmp_path}/data/tables.json", table_key)

        q.user.table_info_path = table_info["schema_info_path"]
        q.user.table_samples_path = table_info["samples_path"]
        q.user.sample_qna_path = table_info["samples_qa"]
        q.user.table_name = table_key

        q.page["select_tables"].table_dropdown.value = table_key
    else:
        q.page["select_tables"].table_dropdown.value = q.user.table_name


async def init(q: Q) -> None:
    q.client.timezone = "UTC"
    username, profile_pic = q.auth.username, q.app.persona_path
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
                items=[ui.nav_item(name="#datasets", label="Upload"), ui.nav_item(name="#chat", label="Chat")],
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
                title=username,
                size="xs",
                image=profile_pic,
            ),
        ],
    )

    # Connect to LLM
    openai.api_key = ""

    await user_variable(q)
    await client_variable(q)
    # If no active hash present, render chat.
    if q.args["#"] is None:
        await chat(q)


def on_shutdown():
    logging.debug("App stopped. Goodbye!")


@app("/", on_shutdown=on_shutdown)
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        setup_dir(base_path)
        await init(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
    await q.page.save()
