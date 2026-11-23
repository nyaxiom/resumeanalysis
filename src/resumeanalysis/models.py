"""ML-only ORM tables. These user IDs refer to the trusted caller's identities."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MLUser(Base):
    __tablename__ = "ml_users"
    __table_args__ = (CheckConstraint("revision >= 0", name="ml_users_revision_nonnegative"),)

    user_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    model_json: Mapped[str | None] = mapped_column(Text)
    model_checksum: Mapped[str | None] = mapped_column(String(64))


class MLVacancy(Base):
    __tablename__ = "ml_vacancies"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("ml_users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    vacancy_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)


class MLFeedback(Base):
    __tablename__ = "ml_feedback"
    __table_args__ = (
        UniqueConstraint("user_id", "vacancy_id", name="ml_feedback_user_vacancy_unique"),
        ForeignKeyConstraint(
            ["user_id", "vacancy_id"],
            ["ml_vacancies.user_id", "ml_vacancies.vacancy_id"],
            ondelete="CASCADE",
            name="ml_feedback_vacancy_fk",
        ),
    )

    sequence: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(200))
    vacancy_id: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    liked: Mapped[bool] = mapped_column(Boolean)
