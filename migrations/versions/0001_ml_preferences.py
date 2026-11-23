"""Create isolated ML preference tables; no changes to ApplyPilot tables."""

import sqlalchemy as sa
from alembic import op

revision = "0001_ml_preferences"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ml_users",
        sa.Column("user_id", sa.String(200), primary_key=True),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("model_json", sa.Text(), nullable=True),
        sa.Column("model_checksum", sa.String(64), nullable=True),
        sa.CheckConstraint("revision >= 0", name="ml_users_revision_nonnegative"),
    )
    op.create_table(
        "ml_vacancies",
        sa.Column("user_id", sa.String(200), primary_key=True),
        sa.Column("vacancy_id", sa.String(200), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["ml_users.user_id"], ondelete="CASCADE"),
    )
    op.create_table(
        "ml_feedback",
        sa.Column("sequence", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("vacancy_id", sa.String(200), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("liked", sa.Boolean(), nullable=False),
        sa.UniqueConstraint("user_id", "vacancy_id", name="ml_feedback_user_vacancy_unique"),
        sa.ForeignKeyConstraint(
            ["user_id", "vacancy_id"],
            ["ml_vacancies.user_id", "ml_vacancies.vacancy_id"],
            name="ml_feedback_vacancy_fk",
            ondelete="CASCADE",
        ),
    )


def downgrade():
    op.drop_table("ml_feedback")
    op.drop_table("ml_vacancies")
    op.drop_table("ml_users")
