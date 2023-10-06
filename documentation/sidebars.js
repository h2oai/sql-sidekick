module.exports = {
  defaultSidebar: [
    "index",
    {
      "Get started": [
        "get-started/what-is-application-name",
        "get-started/access-application-name/access-application-name",
        "get-started/application-name-flow",
        "get-started/use-cases",
        "get-started/videos",
      ],
    },
    {
      type: "category",
      label: "Tutorials",
      link: { type: "doc", id: "tutorials/tutorials-overview" },
      items: [
        {
          type: "category",
          label: "Datasets",
          items: [
            "tutorials/datasets/tutorial-1a",
            "tutorials/datasets/tutorial-2a",
            "tutorials/datasets/tutorial-3a",
          ],
        },
        {
          type: "category",
          label: "Experiments",
          items: [
            "tutorials/experiments/tutorial-1b",
            "tutorials/experiments/tutorial-2b",
            "tutorials/experiments/tutorial-3b",
          ],
        },
        {
          type: "category",
          label: "Predictions",
          items: [
            "tutorials/predictions/tutorial-1c",
            "tutorials/predictions/tutorial-2c",
            "tutorials/predictions/tutorial-3c",
          ],
        },
      ],
    },
    "concepts",
    {
      "User guide": ["user-guide/page-1"],
    },
    {
      "Admin guide": ["admin-guide/page-1"],
    },
    {
      "Python client guide": ["python-client-guide/page-1"],
    },
    "key-terms",
    "release-notes",
    "third-party-licenses",
    "faqs",
  ],
};

