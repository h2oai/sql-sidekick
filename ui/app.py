import gc
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import openai
import toml
import torch
from h2o_wave import Q, app, data, handle_on, main, on, ui
from h2o_wave.core import expando_to_dict
from sidekick.prompter import db_setup_api, query_api
from sidekick.query import SQLGenerator
from sidekick.utils import (MODEL_CHOICE_MAP_DEFAULT,
                            MODEL_CHOICE_MAP_EVAL_MODE, TASK_CHOICE,
                            get_table_keys, save_query, setup_dir,
                            update_tables)

# Load the config file and initialize required paths
app_base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{app_base_path}/ui/app_config.toml")
# Below check is to handle the case when the app is running on the h2o.ai cloud or locally
base_path = app_base_path if os.path.isdir("./.sidekickvenv/bin/") else "/meta_data"
tmp_path = f"{base_path}/var/lib/tmp"

ui_title = env_settings["WAVE_UI"]["TITLE"]
ui_description = env_settings["WAVE_UI"]["SUB_TITLE"]


# Pre-initialize the models for faster response
def initialize_models():
    logging.info(f"Initializing models")

    _ = SQLGenerator(
        None,
        None,
        model_name=None,  # Default: h2ogpt-sql-sqlcoder2
        job_path=base_path,
        data_input_path="",
        sample_queries_path="",
        is_regenerate_with_options="",
        is_regenerate="",
    )
    return


logging.info("Initializing the models")
initialize_models()


async def user_variable(q: Q):
    db_settings = toml.load(f"{app_base_path}/sidekick/configs/env.toml")

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

    q.user.model_choices = MODEL_CHOICE_MAP_DEFAULT
    q.user.eval_mode = False

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

    if q.args.table_dropdown or q.args.model_choice_dropdown or q.args.task_dropdown:
        # If a table/model is selected, the trigger causes refresh of the page
        # so we update chat history with table name selection and return
        # avoiding re-drawing.
        q.page["chat_card"].data += [q.args.chatbot, False]
        return

    if not q.args.chatbot:
        clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    table_names = []
    tables, _ = get_table_keys(f"{tmp_path}/data/tables.json", None)
    if len(tables) > 0:
        with open(f"{tmp_path}/data/tables.json", "r") as json_file:
            meta_data = json.load(json_file)
            for table in tables:
                original_name = meta_data[table].get("original_name", q.user.original_name)
                table_names.append(ui.choice(table, f"{original_name}"))

    MODEL_CHOICE_MAP = q.user.model_choices
    model_choices = [ui.choice(_key, _key) for _key in MODEL_CHOICE_MAP.keys()]
    q.user.model_choice_dropdown = "h2ogpt-sql-sqlcoder2"

    task_choices = [ui.choice("q_a", "Ask Questions"), ui.choice("sqld", "Debugging")]
    q.user.task_choice_dropdown = "q_a"

    chat_card_command_items = [
        ui.command(name="download_accept", label="Download QnA history", icon="Download"),
        ui.command(name="download_reject", label="Download in-correct QnA history", icon="Download"),
    ]

    add_card(
        q,
        "background_card",
        ui.form_card(
            box="horizontal",
            items=[
                ui.text("Ask Questions:"),
                ui.inline(items=[ui.toggle(name="demo_mode", label="Demo", trigger=True)], justify="end"),
            ],
        ),
    ),

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
                    trigger=True,
                ),
                ui.dropdown(
                    name="model_choice_dropdown",
                    label="Model Choice",
                    required=True,
                    choices=model_choices,
                    value=q.user.model_choice_dropdown if q.user.model_choice_dropdown else None,
                    trigger=True,
                ),
            ],
        ),
    ),
    add_card(
        q,
        "task_choice",
        ui.form_card(
            box="vertical",
            items=[
                ui.dropdown(
                    name="task_dropdown",
                    label="Mode",
                    required=True,
                    choices=task_choices,
                    value=q.user.task_choice_dropdown if q.user.task_choice_dropdown else None,
                    trigger=True,
                )
            ],
        ),
    ),
    if not q.args.chatbot:
        add_card(
            q,
            "chat_card",
            ui.chatbot_card(
                box=ui.box("vertical", height="500px"),
                name="chatbot",
                data=data(fields="content from_user", t="list", size=-50),
                commands=[
                    ui.command(name="download_accept", label="Download QnA history", icon="Download"),
                    ui.command(name="download_reject", label="Download in-correct QnA history", icon="Download"),
                ],
                events=["scroll"],
            ),
        ),
    add_card(
        q,
        "additional_actions",
        ui.form_card(
            box=ui.box("vertical", height="120px"),
            items=[
                ui.buttons(
                    [
                        ui.button(
                            name="regenerate",
                            icon="RepeatOne",
                            caption="Attempts regeneration of the last response",
                            label="Try Again",
                            primary=True,
                        ),
                        ui.button(
                            name="regenerate_with_options",
                            icon="RepeatAll",
                            caption="Regenerates with options",
                            label="Try Harder",
                        ),
                        ui.button(
                            name="save_conversation",
                            caption="Saves the conversation in the history for future reference to improve response",
                            label="Accept",
                            icon="Emoji2",
                        ),
                        ui.button(
                            name="save_rejected_conversation",
                            caption="Saves the disappointed conversation to improve response.",
                            label="Reject",
                            icon="EmojiDisappointed",
                        ),
                    ],
                    justify="center",
                )
            ],
        ),
    )

    if q.args.chatbot is None or q.args.chatbot.strip() == "":
        _msg = """Welcome to the SQL Sidekick!\nI am an AI assistant, i am here to help you find answers to questions on structured data.
To get started, please select a table from the dropdown and ask your question.
One could start by learning about the dataset by asking questions like:
- Describe data."""
        q.args.chatbot = _msg
        q.page["chat_card"].data += [q.args.chatbot, False]
    logging.info(f"Chatbot response: {q.args.chatbot}")


@on("chatbot")
async def chatbot(q: Q):
    q.page["sidebar"].value = "#chat"

    # Append user message.
    q.page["chat_card"].data += [q.args.chatbot, True]

    if q.page["select_tables"].table_dropdown.value is None or q.user.table_name is None:
        q.page["chat_card"].data += ["Please select a table to continue!", False]
        return

    if (
        f"Table {q.user.table_dropdown} selected" in q.args.chatbot
        or f"Model {q.user.model_choice_dropdown} selected" in q.args.chatbot
        or f"mode selected" in q.args.chatbot
    ):
        return

    # Append bot response.
    question = f"{q.args.chatbot}"
    # Check on task choice.
    if q.user.task_dropdown == "sqld":
        question = f"Execute SQL:\n{q.args.chatbot}"
    logging.info(f"Question: {question}")

    # For regeneration, currently there are 2 modes
    # 1. Quick fast approach by throttling the temperature
    # 2. "Try harder mode (THM)" Slow approach by using the diverse beam search
    llm_response = None
    try:
        if q.args.chatbot and q.args.chatbot.lower() == "db setup":
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
        elif q.args.chatbot and q.args.chatbot.lower() == "regenerate" or q.args.regenerate:
            # Attempts to regenerate response on the last supplied query
            logging.info(f"Attempt for regeneration")
            if q.client.query is not None and q.client.query.strip() != "":
                llm_response, alt_response, err = query_api(
                    question=q.client.query,
                    sample_queries_path=q.user.sample_qna_path,
                    table_info_path=q.user.table_info_path,
                    table_name=q.user.table_name,
                    model_name=q.user.model_choice_dropdown,
                    is_regenerate=True,
                    is_regen_with_options=False,
                )
                llm_response = "\n".join(llm_response)
            else:
                llm_response = (
                    "Sure, I can generate a new response for you. "
                    "However, in order to assist you effectively could you please provide me with your question?"
                )
        elif q.args.chatbot and q.args.chatbot.lower() == "try harder" or q.args.regenerate_with_options:
            # Attempts to regenerate response on the last supplied query
            logging.info(f"Attempt for regeneration with options.")
            if q.client.query is not None and q.client.query.strip() != "":
                llm_response, alt_response, err = query_api(
                    question=q.client.query,
                    sample_queries_path=q.user.sample_qna_path,
                    table_info_path=q.user.table_info_path,
                    table_name=q.user.table_name,
                    model_name=q.user.model_choice_dropdown,
                    is_regenerate=False,
                    is_regen_with_options=True,
                )
                response = "\n".join(llm_response)
                if alt_response:
                    llm_response = response + "\n\n" + "**Alternate options:**\n" + "\n".join(alt_response)
                    logging.info(f"Regenerate response: {llm_response}")
                else:
                    llm_response = response
            else:
                llm_response = (
                    "Sure, I can generate a new response for you. "
                    "However, in order to assist you effectively could you please provide me with your question?"
                )
        else:
            q.client.query = question
            llm_response, alt_response, err = query_api(
                question=q.client.query,
                sample_queries_path=q.user.sample_qna_path,
                table_info_path=q.user.table_info_path,
                table_name=q.user.table_name,
                model_name=q.user.model_choice_dropdown,
            )
            llm_response = "\n".join(llm_response)
    except (MemoryError, RuntimeError) as e:
        logging.error(f"Something went wrong while generating response: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        llm_response = "Something went wrong, try executing the query again!"
    q.client.llm_response = llm_response
    q.page["chat_card"].data += [llm_response, False]


@on("file_upload")
async def fileupload(q: Q):
    q.page["dataset"].error_bar.visible = False
    q.page["dataset"].error_upload_bar.visible = False
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

    remove_chars = [" ", "-"]
    org_table_name = usr_table_name = None
    if (
        (q.args.table_name == "" or q.args.table_name is None) and sample_data and len(sample_data) > 0
    ):  # User did not provide a table name, use the filename as table name
        org_table_name = sample_data[0].split(".")[0].split("/")[-1]
        logging.info(f"Using provided filename as table name: {org_table_name}")
        q.args.table_name = org_table_name
    if q.args.table_name:
        org_table_name = q.args.table_name
        usr_table_name = org_table_name.strip().lower()
        for _c in remove_chars:
            usr_table_name = usr_table_name.replace(_c, "_")

    logging.info(f"Upload initiated for {org_table_name} with scheme input: {sample_schema}")
    if sample_data is None:
        q.page["dataset"].error_bar.visible = True
        q.page["dataset"].error_upload_bar.visible = False
        q.page["dataset"].progress_bar.visible = False
    else:
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
            "original_name": org_table_name,
            "schema_info_path": usr_info_path,
            "samples_path": usr_samples_path,
            "samples_qa": usr_sample_qa,
        }
        try:
            logging.info(f"Table metadata: {table_metadata}")
            update_tables(f"{tmp_path}/data/tables.json", table_metadata)

            q.user.table_name = usr_table_name
            q.user.table_samples_path = usr_samples_path
            q.user.table_info_path = usr_info_path
            q.user.sample_qna_path = usr_sample_qa

            n_rows, db_resp = db_setup_api(
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
            if "error" in str(db_resp).lower():
                q.page["dataset"].error_upload_bar.visible = True
                q.page["dataset"].error_bar.visible = False
                q.page["dataset"].progress_bar.visible = False
            else:
                q.page["dataset"].progress_bar.visible = False
                q.page["dataset"].success_bar.text = f"Data successfully uploaded, it has {n_rows:,} rows!"
                q.page["dataset"].success_bar.visible = True
        except Exception as e:
            logging.error(f"Something went wrong while uploading the dataset: {e}")
            q.page["dataset"].error_upload_bar.visible = True
            q.page["dataset"].error_bar.visible = False
            q.page["dataset"].progress_bar.visible = False
            return

@on("#settings")
async def on_settings(q: Q):
    q.page["sidebar"].value = "#settings"
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    add_card(q, "settings_header", ui.form_card(box="horizontal", title="Configure", items=[]))

    toggle_state = q.user.eval_mode if q.user.eval_mode else False
    add_card(
        q,
        "dataset",
        ui.form_card(
            box="vertical",
            items=[
                ui.toggle(name='eval_mode', label='Eval Mode', value=toggle_state)]
        ))


@on("#datasets")
async def datasets(q: Q):
    q.page["sidebar"].value = "#datasets"
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    add_card(q, "data_header", ui.form_card(box="horizontal", title="Input Data", items=[]))

    add_card(
        q,
        "dataset",
        ui.form_card(
            box="vertical",
            items=[
                ui.message_bar(
                    name="error_bar",
                    type="error",
                    text="Please input table name and upload data to get started!",
                    visible=False,
                ),
                ui.message_bar(
                    name="error_upload_bar",
                    type="error",
                    text="Upload failed; something went wrong. Please check the dataset name/column name for special characters and try again!",
                    visible=False,
                ),
                ui.message_bar(
                    name="success_bar",
                    type="success",
                    text=f"Data successfully uploaded!",
                    visible=False,
                ),
                ui.file_upload(
                    name="sample_data",
                    label="Dataset",
                    compact=True,
                    multiple=False,
                    file_extensions=["csv"],
                    required=True,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="Upload data to ask questions (currently only .CSV is supported)",
                ),
                ui.separator(label="Optional"),
                ui.textbox(
                    name="table_name",
                    label="Table Name",
                    tooltip="Name of the table to be created, by default data filename is used!",
                ),
                ui.file_upload(
                    name="data_schema",
                    label="Data Schema",
                    multiple=False,
                    compact=True,
                    file_extensions=["jsonl"],
                    max_file_size=5000,  # Specified in MB.
                    tooltip="The schema input summarizing the uploaded structured table, formats allowed are JSONL. If not provided, default schema will be inferred from the data",
                ),
                ui.file_upload(
                    name="sample_qa",
                    label="Sample Q&A",
                    multiple=False,
                    compact=True,
                    file_extensions=["csv"],
                    required=False,
                    max_file_size=5000,  # Specified in MB.
                    tooltip="Sample QnA pairs to improve contextual generation (currently only .CSV is supported)",
                ),
                ui.progress(
                    name="progress_bar", width="100%", label="Uploading datasets and creating tables!", visible=False
                ),
                ui.button(name="file_upload", label="Upload", primary=True),
            ],
        ),
    )


@on("#documentation")
async def about(q: Q):
    q.page["meta"].script = ui.inline_script(f"window.open('https://github.com/h2oai/sql-sidekick');")


@on("#support")
async def on_support(q: Q):
    q.page["meta"].script = ui.inline_script(f"window.open('https://github.com/h2oai/sql-sidekick/issues');")


@on("submit_table")
async def submit_table(q: Q):
    table_key = q.args.table_dropdown
    if table_key:
        table_name = table_key.lower().replace(" ", "_")
        _, table_info = get_table_keys(f"{tmp_path}/data/tables.json", table_name)

        q.user.table_info_path = table_info["schema_info_path"]
        q.user.table_samples_path = table_info["samples_path"]
        q.user.sample_qna_path = table_info["samples_qa"]
        q.user.table_name = table_key.replace(" ", "_")
        q.user.original_name = table_info["original_name"]
        q.page["select_tables"].table_dropdown.value = table_name
    else:
        q.page["select_tables"].table_dropdown.value = q.user.table_name
    await q.page.save()


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
                items=[
                    ui.nav_item(name="#datasets", label="Upload Dataset", icon="Database"),
                    ui.nav_item(name="#chat", label="Chat", icon="Chat"),
                    ui.nav_item(name="#settings", label="Settings", icon="Settings")
                ],
            ),
            ui.nav_group(
                "Help",
                items=[
                    ui.nav_item(name="#documentation", label="Documentation", icon="TextDocument"),
                    ui.nav_item(name="#support", label="Support", icon="Telemarketer"),
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
    logging.info("App stopped. Goodbye!")


# Preload sample data for the app
def upload_demo_examples(q: Q):
    upload_action = True
    cur_dir = os.getcwd()
    sample_data_path = f"{cur_dir}/examples/demo/"
    org_table_name = "Sleep health and lifestyle study"
    usr_table_name = org_table_name.lower().replace(" ", "_")

    table_metadata_path = f"{tmp_path}/data/tables.json"
    # Do not upload dataset if user had any tables uploaded previously. This check avoids re-uploading sample dataset.
    if os.path.exists(table_metadata_path):
        # Read the existing content from the JSON file
        with open(table_metadata_path, "r") as json_file:
            existing_data = json.load(json_file)
            if usr_table_name in existing_data:
                upload_action = False
                logging.info(f"Dataset already uploaded, skipping upload!")

    if upload_action:
        table_metadata = dict()
        table_metadata[usr_table_name] = {
            "original_name": org_table_name,
            "schema_info_path": f"{sample_data_path}/table_info.jsonl",
            "samples_path": f"{sample_data_path}/sleep_health_and_lifestyle_dataset.csv",
            "samples_qa": None,
        }
        update_tables(f"{tmp_path}/data/tables.json", table_metadata)

        q.user.org_table_name = org_table_name
        q.user.table_name = usr_table_name
        q.user.table_samples_path = f"{sample_data_path}/sleep_health_and_lifestyle_dataset.csv"
        q.user.table_info_path = f"{sample_data_path}/table_info.jsonl"
        q.user.sample_qna_path = None

        _, db_resp = db_setup_api(
            db_name=q.user.db_name,
            hostname=q.user.host_name,
            user_name=q.user.user_name,
            password=q.user.password,
            port=q.user.port,
            table_info_path=q.user.table_info_path,
            table_samples_path=q.user.table_samples_path,
            table_name=q.user.table_name,
        )
        logging.info(f"DB updated with demo examples: \n {db_resp}")
    q.args.table_dropdown = usr_table_name


async def on_event(q: Q):
    event_handled = False
    args_dict = expando_to_dict(q.args)
    logging.info(f"Args dict {args_dict}")
    if q.args.regenerate_with_options:
        q.args.chatbot = "try harder"
    elif q.args.regenerate:
        q.args.chatbot = "regenerate"
    q.user.eval_mode  = False

    if q.args.eval_mode:
        q.user.eval_mode = True
        q.user.model_choices = MODEL_CHOICE_MAP_EVAL_MODE
        await chat(q)
        event_handled = True
    if q.args.table_dropdown and not q.args.chatbot and q.user.table_name != q.args.table_dropdown:
        logging.info(f"User selected table: {q.args.table_dropdown}")
        await submit_table(q)
        q.args.chatbot = f"Table {q.args.table_dropdown} selected"
        # Refresh response is triggered when user selects a table via dropdown
        event_handled = True
    if (
        q.args.model_choice_dropdown
        and not q.args.chatbot
    ):
        logging.info(f"User selected model type: {q.args.model_choice_dropdown}")
        q.user.model_choice_dropdown = q.args.model_choice_dropdown
        q.page["select_tables"].model_choice_dropdown.value = q.user.model_choice_dropdown
        q.args.chatbot = f"Model {q.user.model_choice_dropdown} selected"
        # Refresh response is triggered when user selects a table via dropdown
        event_handled = True
    if q.args.task_dropdown and not q.args.chatbot and q.user.task_dropdown != q.args.task_dropdown:
        logging.info(f"User selected task: {q.args.task_dropdown}")
        q.user.task_dropdown = q.args.task_dropdown
        q.page["task_choice"].task_dropdown.value = q.user.task_dropdown
        q.args.chatbot = f"'{TASK_CHOICE[q.user.task_dropdown]}' mode selected"
        # Refresh response is triggered when user selects a table via dropdown
        event_handled = True
    if (
        q.args.save_conversation
        or q.args.save_rejected_conversation
        or (q.args.chatbot and "save the qna pair:" in q.args.chatbot.lower())
    ):
        question = q.client.query
        _val = q.client.llm_response
        # Currently, any manual input by the user is a Question by default
        table_name = q.user.table_name if q.user.table_name else "default"
        _is_invalid = True if q.args.save_rejected_conversation else False
        _msg = (
            "Conversation saved successfully!"
            if not _is_invalid
            else "Sorry, we couldn't get it right, we will try to improve!"
        )
        if (
            question is not None
            and "SELECT" in question
            and (question.lower().startswith("question:") or question.lower().startswith("q:"))
        ):
            _q = question.lower().split("q:")[1].split("r:")[0].strip()
            _r = question.lower().split("r:")[1].strip()
            logging.info(f"Saving conversation for question: {_q} and response: {_r}")
            save_query(base_path, table_name, query=_q, response=_r, is_invalid=_is_invalid)
        elif question is not None and _val is not None and _val.strip() != "":
            logging.info(f"Saving conversation for question: {question} and response: {_val}")
            save_query(base_path, table_name, query=question, response=_val, is_invalid=_is_invalid)
        else:
            _msg = "Sorry, try generating a conversation to save."
        q.page["chat_card"].data += [_msg, False]
        event_handled = True
    elif q.args.download_accept:
        result_path = f"{base_path}/var/lib/tmp/.cache/{q.user.table_name}/history.jsonl"
        # Check if path exists
        # If the model selected is GPT models from openAI then disable download
        # We don't want to use those for further improvements externally.
        if Path(result_path).exists() and "gpt-4" not in q.user.model_choice_dropdown and "gpt-3.5-turbo" not in q.user.model_choice_dropdown:
            logging.info(f"Downloading accepted QnA history for table: {q.user.table_name}")
            (server_path,) = await q.site.upload([result_path])
            q.page["meta"].script = ui.inline_script(f'window.open("{server_path}", "_blank");')
            os.remove(result_path)
            _msg = "Download complete!"
        else:
            _msg = "No history found!"
        q.page["chat_card"].data += [_msg, False]
        event_handled = True
    elif q.args.download_reject and "gpt-4" not in q.user.model_choice_dropdown and "gpt-3.5" not in q.user.model_choice_dropdown:
        logging.info(f"Downloading rejected QnA history for table: {q.user.table_name}")
        result_path = f"{base_path}/var/lib/tmp/.cache/{q.user.table_name}/invalid/history.jsonl"
        if Path(result_path).exists():
            (server_path,) = await q.site.upload([result_path])
            q.page["meta"].script = ui.inline_script(f'window.open("{server_path}", "_blank");')
            os.remove(result_path)
            _msg = "Download complete!"
        else:
            _msg = "No history found!"
        q.page["chat_card"].data += [_msg, False]
        event_handled = True
    elif q.args.regenerate or q.args.regenerate_with_options:
        await chatbot(q)
        event_handled = True
    elif q.args.demo_mode:
        logging.info(f"Switching to demo mode!")
        # If demo datasets are not present, register them.
        upload_demo_examples(q)
        logging.info(f"Demo dataset selected: {q.user.table_name}")
        await submit_table(q)
        sample_qs = """
        Data description: The Sleep Health and Lifestyle Dataset comprises 400 rows and 13 columns,
        covering a wide range of variables related to sleep and daily habits.
        It includes details such as gender, age, occupation, sleep duration, quality of sleep,
        physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps,
        and the presence or absence of sleep disorders\n
        Reference: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset \n
        Example questions:\n
        1. Describe data. Tip: For more detailed insights on the data try AutoInsights on the Cloud marketplace.
        2. What is the average sleep duration for each gender?
        3. How does average sleep duration vary across different age groups?
        4. What are the most common occupations among individuals in the dataset?
        5. What is the average sleep duration for each occupation?
        6. What is the average sleep duration for each age group?
        7. What is the effect of Physical Activity Level on Quality of Sleep?
        """
        q.args.chatbot = (
            f"Demo mode is enabled.\nTry below example questions for the selected data to get started,\n{sample_qs}"
        )
        q.page["chat_card"].data += [q.args.chatbot, False]
        q.args.table_dropdown = None
        q.args.model_choice_dropdown = None
        q.args.task_dropdown = None
        await chat(q)
        event_handled = True
    else:  # default chatbot event
        await handle_on(q)
        event_handled = True
    logging.info(f"Event handled: {event_handled} ... ")
    return event_handled


@app("/", on_shutdown=on_shutdown)
async def serve(q: Q):
    # Run only once per client connection
    if not q.client.initialized:
        q.client.cards = set()
        setup_dir(base_path)
        await init(q)
        q.client.initialized = True
        logging.info("App initialized.")

    # Handle routing.
    if await on_event(q):
        await q.page.save()
        return
