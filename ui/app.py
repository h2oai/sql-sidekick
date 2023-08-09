import logging
import json
from pathlib import Path
from typing import List, Optional

import openai
import toml
from h2o_wave import Q, app, data, handle_on, main, on, ui
from sidekick.prompter import db_setup_api, query_api

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/ui/.app_config.toml")
tmp_path = f"{base_path}/var/lib/tmp"

ui_title = env_settings["WAVE_UI"]["TITLE"]
ui_description = env_settings["WAVE_UI"]["SUB_TITLE"]

async def user_variable(q:Q):
    db_settings = toml.load(f"{base_path}/sidekick/configs/.env.toml")

    q.user.db_dialect = db_settings["DB-DIALECT"]["DB_TYPE"]
    q.user.host_name = db_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
    q.user.user_name = db_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
    q.user.password = db_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
    q.user.db_name = db_settings["LOCAL_DB_CONFIG"]["DB_NAME"]
    q.user.port = db_settings["LOCAL_DB_CONFIG"]["PORT"]

    q.user.table_info_path = db_settings["TABLE_INFO"]["TABLE_INFO_PATH"]
    q.user.table_samples_path = db_settings["TABLE_INFO"]["TABLE_SAMPLES_PATH"]
    q.user.sample_qna_path = db_settings["TABLE_INFO"]["SAMPLE_QNA_PATH"]
    q.user.table_name = db_settings["TABLE_INFO"]["TABLE_NAME"]

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
        llm_response = db_setup_api(
            db_name=q.user.db_name,
            hostname=q.user.host_name,
            user_name=q.user.user_name,
            password=q.user.password,
            port=q.user.port,
            table_info_path=q.user.table_info_path,
            table_samples_path=q.user.table_samples_path,
            table_name=q.user.table_name
        )
    else:
        llm_response = query_api(question=question,
                                 sample_queries_path=q.user.sample_qna_path,
                                 table_info_path=q.user.table_info_path)
        llm_response = "\n".join(llm_response)

    q.page["chat_card"].data += [llm_response, False]

@on("file_upload")
async def fileupload(q: Q):
    q.page['dataset'].error_bar.visible = False
    q.page['dataset'].success_bar.visible = False

    q.page["sidebar"].value = "#datasets"
    usr_info_path = None
    usr_samples_path = None
    usr_sample_qa = None

    sample_data = q.args.sample_data
    sample_schema = q.args.data_schema
    sample_qa = q.args.sample_qa

    usr_table_name = q.args.table_name
    # file_type = "csv" if q.args.file_type == "data_samples" else "jsonl"

    if sample_data:
        usr_samples_path = await q.site.download(sample_data[0], f"{tmp_path}/jobs/{usr_table_name}_table_samples.csv")
    if sample_schema:
        usr_info_path = await q.site.download(sample_schema[0], f"{tmp_path}/jobs/{usr_table_name}_table_info.jsonl")
    if sample_qa:
        usr_sample_qa = await q.site.download(sample_qa[0], f"{tmp_path}/jobs/{usr_table_name}_sample_qa.csv")

    if sample_data is None or sample_schema is None:
        q.page['dataset'].error_bar.visible = True
    else:
        q.page['dataset'].error_bar.visible = False
        if Path(f"{tmp_path}/data/tables.json").exists():
            f = open(f"{tmp_path}/data/tables.json", "r")
            try:
                table_metadata = json.load(f)
                f.close()
            except json.JSONDecodeError as e:
                table_metadata = dict()

            table_metadata[usr_table_name] = {"schema_info_path":usr_info_path,
                                          "samples_path": usr_samples_path,
                                          "samples_qa": usr_sample_qa}

            with open(f"{tmp_path}/data/tables.json", "w") as outfile:
                json.dump(table_metadata, outfile, indent=4, sort_keys=False)


            q.user.table_name = usr_table_name
            q.user.table_samples_path = usr_samples_path
            q.user.table_info_path = usr_info_path

            db_resp = db_setup_api(
                db_name=q.user.db_name,
                hostname=q.user.host_name,
                user_name=q.user.user_name,
                password=q.user.password,
                port=q.user.port,
                table_info_path=q.user.table_info_path,
                table_samples_path=q.user.table_samples_path,
                table_name=q.user.table_name
            )
            logging.info(f"DB updates: \n {db_resp}")
            q.page['dataset'].success_bar.visible = True

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
                ui.message_bar(name='error_bar', type='error', text='Please select data & schema files to upload!', visible=False),
                ui.message_bar(name='success_bar', type='success', text='Files Uploaded Successfully!', visible=False),
                ui.textbox(name='table_name', label='Table Name', required=True, value='demo'),
                ui.file_upload(
                    name='sample_data',
                    label='Data Samples',
                    multiple=False,
                    compact=True,
                    file_extensions=['csv'],
                    required=True,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!"
                ),
                ui.file_upload(
                    name='data_schema',
                    label='Data Schema',
                    multiple=False,
                    compact=True,
                    file_extensions=['jsonl'],
                    required=True,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!"
                ),
                ui.file_upload(
                    name='sample_qa',
                    label='Sample Q&A',
                    multiple=False,
                    compact=True,
                    file_extensions=['csv'],
                    required=False,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The data describing table schema and sample values, formats allowed are JSONL & CSV respectively!"
                ),
                ui.button(name='file_upload', label='Submit', primary=True)
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
                    ui.nav_item(name="#datasets", label="Upload"),
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

    await user_variable(q)

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
