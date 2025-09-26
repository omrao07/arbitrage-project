/**
 * db/memory.ts
 * Simple in-memory store for demo / dev usage.
 * Replace with real DB (Postgres, Mongo, etc.) in production.
 */



// User type definition

export type User = {
  id: string;
  email: string;
  name?: string;
  createdAt: string;
  updatedAt: string;
};

const users = new Map<string, User>();

export const MemoryDB = {
  listUsers(): User[] {
    return Array.from(users.values());
  },

  getUser(id: string): User | undefined {
    return users.get(id);
  },

  findByEmail(email: string): User | undefined {
    return Array.from(users.values()).find((u) => u.email === email);
  },

  createUser(email: string, name?: string): User {
    const now = new Date().toISOString();
    const user: User = {
        id: (Math.random() * 1e16).toString(36),
      email,
      name,
      createdAt: now,
      updatedAt: now,
    };
    users.set(user.id, user);
    return user;
  },

  updateUser(id: string, patch: Partial<User>): User | null {
    const u = users.get(id);
    if (!u) return null;
    const next: User = { ...u, ...patch, updatedAt: new Date().toISOString() };
    users.set(id, next);
    return next;
  },

  deleteUser(id: string): boolean {
    return users.delete(id);
  },

  clear(): void {
    users.clear();
  },
};