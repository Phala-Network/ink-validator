//! This module takes care of loading, checking and preprocessing of a
//! wasm module before execution. It also extracts some essential information
//! from a module.

use alloc::vec::Vec;

use anyhow::{anyhow, bail, Result};
use core::fmt::Display;
use wasm_instrument::{
    gas_metering,
    parity_wasm::elements::{self, External, Internal, MemoryType, Type, ValueType},
};
use wasmparser::{Validator, WasmFeatures};

/// Imported memory must be located inside this module. The reason for hardcoding is that current
/// compiler toolchains might not support specifying other modules than "env" for memory imports.
pub const IMPORT_MODULE_MEMORY: &str = "env";

trait ErrContext {
    type T;
    fn context(self, ctx: impl Display) -> Result<Self::T>;
}

impl<T, E: Display> ErrContext for Result<T, E> {
    type T = T;
    fn context(self, ctx: impl Display) -> Result<Self::T> {
        self.map_err(|err| anyhow!("{}: {}", ctx, err))
    }
}

pub struct Schedule {
    pub limits: Limits,
    pub instruction_weights: InstructionWeights,
    pub host_fn_weights: HostFnWeights,
}

impl Default for Schedule {
    fn default() -> Self {
        let mut instruction_weights = InstructionWeights::default();
        let mut limits = Limits::default();

        const MB: u32 = 16;
        limits.memory_pages = 4 * MB;
        instruction_weights.fallback = 1;

        Self {
            limits,
            instruction_weights,
            host_fn_weights: Default::default(),
        }
    }
}

pub struct Limits {
    /// The maximum number of topics supported by an event.
    pub event_topics: u32,

    /// Maximum allowed stack height in number of elements.
    ///
    /// See <https://wiki.parity.io/WebAssembly-StackHeight> to find out
    /// how the stack frame cost is calculated. Each element can be of one of the
    /// wasm value types. This means the maximum size per element is 64bit.
    ///
    /// # Note
    ///
    /// It is safe to disable (pass `None`) the `stack_height` when the execution engine
    /// is part of the runtime and hence there can be no indeterminism between different
    /// client resident execution engines.
    pub stack_height: Option<u32>,

    /// Maximum number of globals a module is allowed to declare.
    ///
    /// Globals are not limited through the `stack_height` as locals are. Neither does
    /// the linear memory limit `memory_pages` applies to them.
    pub globals: u32,

    /// Maximum number of locals a function can have.
    ///
    /// As wasm engine initializes each of the local, we need to limit their number to confine
    /// execution costs.
    pub locals: u32,

    /// Maximum numbers of parameters a function can have.
    ///
    /// Those need to be limited to prevent a potentially exploitable interaction with
    /// the stack height instrumentation: The costs of executing the stack height
    /// instrumentation for an indirectly called function scales linearly with the amount
    /// of parameters of this function. Because the stack height instrumentation itself is
    /// is not weight metered its costs must be static (via this limit) and included in
    /// the costs of the instructions that cause them (call, call_indirect).
    pub parameters: u32,

    /// Maximum number of memory pages allowed for a contract.
    pub memory_pages: u32,

    /// Maximum number of elements allowed in a table.
    ///
    /// Currently, the only type of element that is allowed in a table is funcref.
    pub table_size: u32,

    /// Maximum number of elements that can appear as immediate value to the br_table instruction.
    pub br_table_size: u32,

    /// The maximum length of a subject in bytes used for PRNG generation.
    pub subject_len: u32,

    /// The maximum nesting level of the call stack.
    pub call_depth: u32,

    /// The maximum size of a storage value and event payload in bytes.
    pub payload_len: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            event_topics: 4,
            // No stack limit required because we use a runtime resident execution engine.
            stack_height: None,
            globals: 256,
            locals: 1024,
            parameters: 128,
            memory_pages: 16,
            // 4k function pointers (This is in count not bytes).
            table_size: 4096,
            br_table_size: 256,
            subject_len: 32,
            call_depth: 32,
            payload_len: 16 * 1024,
        }
    }
}

#[derive(Default)]
pub struct InstructionWeights {
    /// Version of the instruction weights.
    ///
    /// # Note
    ///
    /// Should be incremented whenever any instruction weight is changed. The
    /// reason is that changes to instruction weights require a re-instrumentation
    /// in order to apply the changes to an already deployed code. The re-instrumentation
    /// is triggered by comparing the version of the current schedule with the version the code was
    /// instrumented with. Changes usually happen when pallet_contracts is re-benchmarked.
    ///
    /// Changes to other parts of the schedule should not increment the version in
    /// order to avoid unnecessary re-instrumentations.
    pub version: u32,
    /// Weight to be used for instructions which don't have benchmarks assigned.
    ///
    /// This weight is used whenever a code is uploaded with [`Determinism::AllowIndeterminism`]
    /// and an instruction (usually a float instruction) is encountered. This weight is **not**
    /// used if a contract is uploaded with [`Determinism::Deterministic`]. If this field is set to
    /// `0` (the default) only deterministic codes are allowed to be uploaded.
    pub fallback: u32,
    pub i64const: u32,
    pub i64load: u32,
    pub i64store: u32,
    pub select: u32,
    pub r#if: u32,
    pub br: u32,
    pub br_if: u32,
    pub br_table: u32,
    pub br_table_per_entry: u32,
    pub call: u32,
    pub call_indirect: u32,
    pub call_indirect_per_param: u32,
    pub call_per_local: u32,
    pub local_get: u32,
    pub local_set: u32,
    pub local_tee: u32,
    pub global_get: u32,
    pub global_set: u32,
    pub memory_current: u32,
    pub memory_grow: u32,
    pub i64clz: u32,
    pub i64ctz: u32,
    pub i64popcnt: u32,
    pub i64eqz: u32,
    pub i64extendsi32: u32,
    pub i64extendui32: u32,
    pub i32wrapi64: u32,
    pub i64eq: u32,
    pub i64ne: u32,
    pub i64lts: u32,
    pub i64ltu: u32,
    pub i64gts: u32,
    pub i64gtu: u32,
    pub i64les: u32,
    pub i64leu: u32,
    pub i64ges: u32,
    pub i64geu: u32,
    pub i64add: u32,
    pub i64sub: u32,
    pub i64mul: u32,
    pub i64divs: u32,
    pub i64divu: u32,
    pub i64rems: u32,
    pub i64remu: u32,
    pub i64and: u32,
    pub i64or: u32,
    pub i64xor: u32,
    pub i64shl: u32,
    pub i64shrs: u32,
    pub i64shru: u32,
    pub i64rotl: u32,
    pub i64rotr: u32,
}

#[derive(Default)]
pub struct HostFnWeights {
    /// Weight of calling `seal_caller`.
    pub caller: u64,

    /// Weight of calling `seal_is_contract`.
    pub is_contract: u64,

    /// Weight of calling `seal_code_hash`.
    pub code_hash: u64,

    /// Weight of calling `seal_own_code_hash`.
    pub own_code_hash: u64,

    /// Weight of calling `seal_caller_is_origin`.
    pub caller_is_origin: u64,

    /// Weight of calling `seal_address`.
    pub address: u64,

    /// Weight of calling `seal_gas_left`.
    pub gas_left: u64,

    /// Weight of calling `seal_balance`.
    pub balance: u64,

    /// Weight of calling `seal_value_transferred`.
    pub value_transferred: u64,

    /// Weight of calling `seal_minimum_balance`.
    pub minimum_balance: u64,

    /// Weight of calling `seal_block_number`.
    pub block_number: u64,

    /// Weight of calling `seal_now`.
    pub now: u64,

    /// Weight of calling `seal_weight_to_fee`.
    pub weight_to_fee: u64,

    /// Weight of calling `gas`.
    pub gas: u64,

    /// Weight of calling `seal_input`.
    pub input: u64,

    /// Weight per input byte copied to contract memory by `seal_input`.
    pub input_per_byte: u64,

    /// Weight of calling `seal_return`.
    pub r#return: u64,

    /// Weight per byte returned through `seal_return`.
    pub return_per_byte: u64,

    /// Weight of calling `seal_terminate`.
    pub terminate: u64,

    /// Weight of calling `seal_random`.
    pub random: u64,

    /// Weight of calling `seal_reposit_event`.
    pub deposit_event: u64,

    /// Weight per topic supplied to `seal_deposit_event`.
    pub deposit_event_per_topic: u64,

    /// Weight per byte of an event deposited through `seal_deposit_event`.
    pub deposit_event_per_byte: u64,

    /// Weight of calling `seal_debug_message`.
    pub debug_message: u64,

    /// Weight of calling `seal_set_storage`.
    pub set_storage: u64,

    /// Weight per written byten of an item stored with `seal_set_storage`.
    pub set_storage_per_new_byte: u64,

    /// Weight per overwritten byte of an item stored with `seal_set_storage`.
    pub set_storage_per_old_byte: u64,

    /// Weight of calling `seal_set_code_hash`.
    pub set_code_hash: u64,

    /// Weight of calling `seal_clear_storage`.
    pub clear_storage: u64,

    /// Weight of calling `seal_clear_storage` per byte of the stored item.
    pub clear_storage_per_byte: u64,

    /// Weight of calling `seal_contains_storage`.
    pub contains_storage: u64,

    /// Weight of calling `seal_contains_storage` per byte of the stored item.
    pub contains_storage_per_byte: u64,

    /// Weight of calling `seal_get_storage`.
    pub get_storage: u64,

    /// Weight per byte of an item received via `seal_get_storage`.
    pub get_storage_per_byte: u64,

    /// Weight of calling `seal_take_storage`.
    pub take_storage: u64,

    /// Weight per byte of an item received via `seal_take_storage`.
    pub take_storage_per_byte: u64,

    /// Weight of calling `seal_transfer`.
    pub transfer: u64,

    /// Weight of calling `seal_call`.
    pub call: u64,

    /// Weight of calling `seal_delegate_call`.
    pub delegate_call: u64,

    /// Weight surcharge that is claimed if `seal_call` does a balance transfer.
    pub call_transfer_surcharge: u64,

    /// Weight per byte that is cloned by supplying the `CLONE_INPUT` flag.
    pub call_per_cloned_byte: u64,

    /// Weight of calling `seal_instantiate`.
    pub instantiate: u64,

    /// Weight surcharge that is claimed if `seal_instantiate` does a balance transfer.
    pub instantiate_transfer_surcharge: u64,

    /// Weight per salt byte supplied to `seal_instantiate`.
    pub instantiate_per_salt_byte: u64,

    /// Weight of calling `seal_hash_sha_256`.
    pub hash_sha2_256: u64,

    /// Weight per byte hashed by `seal_hash_sha_256`.
    pub hash_sha2_256_per_byte: u64,

    /// Weight of calling `seal_hash_keccak_256`.
    pub hash_keccak_256: u64,

    /// Weight per byte hashed by `seal_hash_keccak_256`.
    pub hash_keccak_256_per_byte: u64,

    /// Weight of calling `seal_hash_blake2_256`.
    pub hash_blake2_256: u64,

    /// Weight per byte hashed by `seal_hash_blake2_256`.
    pub hash_blake2_256_per_byte: u64,

    /// Weight of calling `seal_hash_blake2_128`.
    pub hash_blake2_128: u64,

    /// Weight per byte hashed by `seal_hash_blake2_128`.
    pub hash_blake2_128_per_byte: u64,

    /// Weight of calling `seal_ecdsa_recover`.
    pub ecdsa_recover: u64,

    /// Weight of calling `seal_ecdsa_to_eth_address`.
    pub ecdsa_to_eth_address: u64,

    /// Weight of calling `reentrance_count`.
    pub reentrance_count: u64,

    /// Weight of calling `account_reentrance_count`.
    pub account_reentrance_count: u64,

    /// Weight of calling `instantiation_nonce`.
    pub instantiation_nonce: u64,
}

impl Schedule {
    pub(crate) fn rules(
        &self,
        module: &elements::Module,
        deterministic: bool,
    ) -> impl gas_metering::Rules + '_ {
        ScheduleRules {
            schedule: self,
            params: module
                .type_section()
                .iter()
                .flat_map(|section| section.types())
                .map(|func| {
                    let elements::Type::Function(func) = func;
                    func.params().len() as u32
                })
                .collect(),
            deterministic,
        }
    }
}

struct ScheduleRules<'a> {
    schedule: &'a Schedule,
    params: Vec<u32>,
    deterministic: bool,
}

impl<'a> gas_metering::Rules for ScheduleRules<'a> {
    fn instruction_cost(&self, instruction: &elements::Instruction) -> Option<u32> {
        use self::elements::Instruction::*;
        let w = &self.schedule.instruction_weights;
        let max_params = self.schedule.limits.parameters;

        let weight = match *instruction {
            End | Unreachable | Return | Else => 0,
            I32Const(_) | I64Const(_) | Block(_) | Loop(_) | Nop | Drop => w.i64const,
            I32Load(_, _)
            | I32Load8S(_, _)
            | I32Load8U(_, _)
            | I32Load16S(_, _)
            | I32Load16U(_, _)
            | I64Load(_, _)
            | I64Load8S(_, _)
            | I64Load8U(_, _)
            | I64Load16S(_, _)
            | I64Load16U(_, _)
            | I64Load32S(_, _)
            | I64Load32U(_, _) => w.i64load,
            I32Store(_, _)
            | I32Store8(_, _)
            | I32Store16(_, _)
            | I64Store(_, _)
            | I64Store8(_, _)
            | I64Store16(_, _)
            | I64Store32(_, _) => w.i64store,
            Select => w.select,
            If(_) => w.r#if,
            Br(_) => w.br,
            BrIf(_) => w.br_if,
            Call(_) => w.call,
            GetLocal(_) => w.local_get,
            SetLocal(_) => w.local_set,
            TeeLocal(_) => w.local_tee,
            GetGlobal(_) => w.global_get,
            SetGlobal(_) => w.global_set,
            CurrentMemory(_) => w.memory_current,
            GrowMemory(_) => w.memory_grow,
            CallIndirect(idx, _) => *self.params.get(idx as usize).unwrap_or(&max_params),
            BrTable(ref data) => w
                .br_table
                .saturating_add(w.br_table_per_entry.saturating_mul(data.table.len() as u32)),
            I32Clz | I64Clz => w.i64clz,
            I32Ctz | I64Ctz => w.i64ctz,
            I32Popcnt | I64Popcnt => w.i64popcnt,
            I32Eqz | I64Eqz => w.i64eqz,
            I64ExtendSI32 => w.i64extendsi32,
            I64ExtendUI32 => w.i64extendui32,
            I32WrapI64 => w.i32wrapi64,
            I32Eq | I64Eq => w.i64eq,
            I32Ne | I64Ne => w.i64ne,
            I32LtS | I64LtS => w.i64lts,
            I32LtU | I64LtU => w.i64ltu,
            I32GtS | I64GtS => w.i64gts,
            I32GtU | I64GtU => w.i64gtu,
            I32LeS | I64LeS => w.i64les,
            I32LeU | I64LeU => w.i64leu,
            I32GeS | I64GeS => w.i64ges,
            I32GeU | I64GeU => w.i64geu,
            I32Add | I64Add => w.i64add,
            I32Sub | I64Sub => w.i64sub,
            I32Mul | I64Mul => w.i64mul,
            I32DivS | I64DivS => w.i64divs,
            I32DivU | I64DivU => w.i64divu,
            I32RemS | I64RemS => w.i64rems,
            I32RemU | I64RemU => w.i64remu,
            I32And | I64And => w.i64and,
            I32Or | I64Or => w.i64or,
            I32Xor | I64Xor => w.i64xor,
            I32Shl | I64Shl => w.i64shl,
            I32ShrS | I64ShrS => w.i64shrs,
            I32ShrU | I64ShrU => w.i64shru,
            I32Rotl | I64Rotl => w.i64rotl,
            I32Rotr | I64Rotr => w.i64rotr,
            _ if !self.deterministic && w.fallback > 0 => w.fallback,
            _ => return None,
        };
        Some(weight)
    }

    fn memory_grow_cost(&self) -> gas_metering::MemoryGrowCost {
        // We benchmarked the memory.grow instruction with the maximum allowed pages.
        // The cost for growing is therefore already included in the instruction cost.
        gas_metering::MemoryGrowCost::Free
    }

    fn call_per_local_cost(&self) -> u32 {
        self.schedule.instruction_weights.call_per_local
    }
}

struct ContractModule<'a> {
    /// A deserialized module. The module is valid (this is Guaranteed by `new` method).
    module: elements::Module,
    schedule: &'a Schedule,
}

impl<'a> ContractModule<'a> {
    /// Creates a new instance of `ContractModule`.
    ///
    /// Returns `Err` if the `original_code` couldn't be decoded or
    /// if it contains an invalid module.
    fn new(original_code: &[u8], schedule: &'a Schedule) -> Result<Self> {
        let module =
            elements::deserialize_buffer(original_code).context("Can't decode wasm code")?;

        // Return a `ContractModule` instance with
        // __valid__ module.
        Ok(ContractModule { module, schedule })
    }

    /// Ensures that module doesn't declare internal memories.
    ///
    /// In this runtime we only allow wasm module to import memory from the environment.
    /// Memory section contains declarations of internal linear memories, so if we find one
    /// we reject such a module.
    fn ensure_no_internal_memory(&self) -> Result<()> {
        if self
            .module
            .memory_section()
            .map_or(false, |ms| ms.entries().len() > 0)
        {
            bail!("module declares internal memory");
        }
        Ok(())
    }

    /// Ensures that tables declared in the module are not too big.
    fn ensure_table_size_limit(&self, limit: u32) -> Result<()> {
        if let Some(table_section) = self.module.table_section() {
            // In Wasm MVP spec, there may be at most one table declared. Double check this
            // explicitly just in case the Wasm version changes.
            if table_section.entries().len() > 1 {
                bail!("multiple tables declared");
            }
            if let Some(table_type) = table_section.entries().first() {
                // Check the table's initial size as there is no instruction or environment function
                // capable of growing the table.
                if table_type.limits().initial() > limit {
                    bail!("table exceeds maximum size allowed");
                }
            }
        }
        Ok(())
    }

    /// Ensure that any `br_table` instruction adheres to its immediate value limit.
    fn ensure_br_table_size_limit(&self, limit: u32) -> Result<()> {
        let code_section = if let Some(type_section) = self.module.code_section() {
            type_section
        } else {
            return Ok(());
        };
        for instr in code_section
            .bodies()
            .iter()
            .flat_map(|body| body.code().elements())
        {
            use self::elements::Instruction::BrTable;
            if let BrTable(table) = instr {
                if table.table.len() > limit as usize {
                    bail!("BrTable's immediate value is too big.");
                }
            }
        }
        Ok(())
    }

    fn ensure_global_variable_limit(&self, limit: u32) -> Result<()> {
        if let Some(global_section) = self.module.global_section() {
            if global_section.entries().len() > limit as usize {
                bail!("module declares too many globals");
            }
        }
        Ok(())
    }

    fn ensure_local_variable_limit(&self, limit: u32) -> Result<()> {
        if let Some(code_section) = self.module.code_section() {
            for func_body in code_section.bodies() {
                let locals_count: u32 = func_body
                    .locals()
                    .iter()
                    .map(|val_type| val_type.count())
                    .sum();
                if locals_count > limit {
                    bail!("single function declares too many locals");
                }
            }
        }
        Ok(())
    }

    /// Ensures that no floating point types are in use.
    fn ensure_no_floating_types(&self) -> Result<()> {
        if let Some(global_section) = self.module.global_section() {
            for global in global_section.entries() {
                match global.global_type().content_type() {
                    ValueType::F32 | ValueType::F64 => {
                        bail!("use of floating point type in globals is forbidden")
                    }
                    _ => {}
                }
            }
        }

        if let Some(code_section) = self.module.code_section() {
            for func_body in code_section.bodies() {
                for local in func_body.locals() {
                    match local.value_type() {
                        ValueType::F32 | ValueType::F64 => {
                            bail!("use of floating point type in locals is forbidden")
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(type_section) = self.module.type_section() {
            for wasm_type in type_section.types() {
                match wasm_type {
                    Type::Function(func_type) => {
                        let return_type = func_type.results().get(0);
                        for value_type in func_type.params().iter().chain(return_type) {
                            match value_type {
                                ValueType::F32 | ValueType::F64 => {
                                    bail!(
                                        "use of floating point type in function types is forbidden",
                                    )
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Ensure that no function exists that has more parameters than allowed.
    fn ensure_parameter_limit(&self, limit: u32) -> Result<()> {
        let type_section = if let Some(type_section) = self.module.type_section() {
            type_section
        } else {
            return Ok(());
        };

        for Type::Function(func) in type_section.types() {
            if func.params().len() > limit as usize {
                bail!("Use of a function type with too many parameters.");
            }
        }

        Ok(())
    }

    fn inject_gas_metering(self, deterministic: bool) -> Result<Self> {
        let gas_rules = self.schedule.rules(&self.module, deterministic);
        let backend = gas_metering::host_function::Injector::new("seal0", "gas");
        // TODO
        let contract_module = gas_metering::inject(self.module, backend, &gas_rules)
            .map_err(|_| anyhow!("gas instrumentation failed"))?;
        Ok(ContractModule {
            module: contract_module,
            schedule: self.schedule,
        })
    }

    fn inject_stack_height_metering(self) -> Result<Self> {
        if let Some(limit) = self.schedule.limits.stack_height {
            let contract_module = wasm_instrument::inject_stack_limiter(self.module, limit)
                .context("stack height instrumentation failed")?;
            Ok(ContractModule {
                module: contract_module,
                schedule: self.schedule,
            })
        } else {
            Ok(ContractModule {
                module: self.module,
                schedule: self.schedule,
            })
        }
    }

    /// Check that the module has required exported functions. For now
    /// these are just entrypoints:
    ///
    /// - 'call'
    /// - 'deploy'
    ///
    /// Any other exports are not allowed.
    fn scan_exports(&self) -> Result<()> {
        let mut deploy_found = false;
        let mut call_found = false;

        let module = &self.module;

        let types = module.type_section().map(|ts| ts.types()).unwrap_or(&[]);
        let export_entries = module
            .export_section()
            .map(|is| is.entries())
            .unwrap_or(&[]);
        let func_entries = module
            .function_section()
            .map(|fs| fs.entries())
            .unwrap_or(&[]);

        // Function index space consists of imported function following by
        // declared functions. Calculate the total number of imported functions so
        // we can use it to convert indexes from function space to declared function space.
        let fn_space_offset = module
            .import_section()
            .map(|is| is.entries())
            .unwrap_or(&[])
            .iter()
            .filter(|entry| matches!(*entry.external(), External::Function(_)))
            .count();

        for export in export_entries {
            match export.field() {
                "call" => call_found = true,
                "deploy" => deploy_found = true,
                _ => bail!("unknown export: expecting only deploy and call functions"),
            }

            // Then check the export kind. "call" and "deploy" are
            // functions.
            let fn_idx = match export.internal() {
                Internal::Function(ref fn_idx) => *fn_idx,
                _ => bail!("expected call and deploy to be a function"),
            };

            // convert index from function index space to declared index space.
            let fn_idx = match fn_idx.checked_sub(fn_space_offset as u32) {
                Some(fn_idx) => fn_idx,
                None => {
                    // Underflow here means fn_idx points to imported function which we don't allow!
                    bail!("entry point points to an imported function");
                }
            };

            // Then check the signature.
            // Both "call" and "deploy" has a () -> () function type.
            // We still support () -> (i32) for backwards compatibility.
            let func_ty_idx = func_entries
                .get(fn_idx as usize)
                .ok_or(anyhow!("export refers to non-existent function"))?
                .type_ref();
            let Type::Function(ref func_ty) = types
                .get(func_ty_idx as usize)
                .ok_or(anyhow!("function has a non-existent type"))?;
            if !(func_ty.params().is_empty()
                && (func_ty.results().is_empty() || func_ty.results() == [ValueType::I32]))
            {
                bail!("entry point has wrong signature");
            }
        }

        if !deploy_found {
            bail!("deploy function isn't exported");
        }
        if !call_found {
            bail!("call function isn't exported");
        }

        Ok(())
    }

    /// Scan an import section if any.
    ///
    /// This makes sure that the import section looks as we expect it from a contract
    /// and enforces and returns the memory type declared by the contract if any.
    ///
    /// `import_fn_banlist`: list of function names that are disallowed to be imported
    fn scan_imports(&self, import_fn_banlist: &[&[u8]]) -> Result<Option<&MemoryType>> {
        let module = &self.module;
        let import_entries = module
            .import_section()
            .map(|is| is.entries())
            .unwrap_or(&[]);
        let mut imported_mem_type = None;

        for import in import_entries {
            match *import.external() {
                External::Table(_) => bail!("Cannot import tables"),
                External::Global(_) => bail!("Cannot import globals"),
                External::Function(_) => {
                    if import_fn_banlist
                        .iter()
                        .any(|f| import.field().as_bytes() == *f)
                    {
                        bail!("module imports a banned function: {}", import.field());
                    }
                }
                External::Memory(ref memory_type) => {
                    if import.module() != IMPORT_MODULE_MEMORY {
                        bail!("Invalid module for imported memory");
                    }
                    if import.field() != "memory" {
                        bail!("Memory import must have the field name 'memory'");
                    }
                    if imported_mem_type.is_some() {
                        bail!("Multiple memory imports defined");
                    }
                    imported_mem_type = Some(memory_type);
                    continue;
                }
            }
        }
        Ok(imported_mem_type)
    }

    fn into_wasm_code(self) -> Result<Vec<u8>> {
        elements::serialize(self.module).context("error serializing instrumented module")
    }
}

fn get_memory_limits(module: Option<&MemoryType>, schedule: &Schedule) -> Result<(u32, u32)> {
    if let Some(memory_type) = module {
        // Inspect the module to extract the initial and maximum page count.
        let limits = memory_type.limits();
        match (limits.initial(), limits.maximum()) {
            (initial, Some(maximum)) if initial > maximum => {
                bail!("Requested initial number of pages should not exceed the requested maximum")
            }
            (_, Some(maximum)) if maximum > schedule.limits.memory_pages => {
                bail!("Maximum number of pages should not exceed the configured maximum.")
            }
            (initial, Some(maximum)) => Ok((initial, maximum)),
            (_, None) => {
                // Maximum number of pages should be always declared.
                // This isn't a hard requirement and can be treated as a maximum set
                // to configured maximum.
                bail!("Maximum number of pages should be always declared.")
            }
        }
    } else {
        // If none memory imported then just crate an empty placeholder.
        // Any access to it will lead to out of bounds trap.
        Ok((0, 0))
    }
}

/// Check and instrument the given `original_code`.
///
/// On success it returns the instrumented versions together with its `(initial, maximum)`
/// error requirement. The memory requirement was also validated against the `schedule`.
pub fn instrument(original_code: &[u8], deterministic: bool) -> Result<(Vec<u8>, (u32, u32))> {
    let schedule = &Schedule::default();
    // Do not enable any features here. Any additional feature needs to be carefully
    // checked for potential security issues. For example, enabling multi value could lead
    // to a DoS vector: It breaks our assumption that branch instructions are of constant time.
    // Depending on the implementation they can linearly depend on the amount of values returned
    // from a block.
    Validator::new_with_features(WasmFeatures {
        relaxed_simd: false,
        threads: false,
        tail_call: false,
        multi_memory: false,
        exceptions: false,
        memory64: false,
        extended_const: false,
        component_model: false,
        deterministic_only: deterministic,
        mutable_global: false,
        saturating_float_to_int: false,
        sign_extension: false,
        bulk_memory: false,
        multi_value: false,
        reference_types: false,
        simd: false,
    })
    .validate_all(original_code)
    .context("validation of new code failed")?;

    let contract_module = ContractModule::new(original_code, schedule)?;
    contract_module.scan_exports()?;
    contract_module.ensure_no_internal_memory()?;
    contract_module.ensure_table_size_limit(schedule.limits.table_size)?;
    contract_module.ensure_global_variable_limit(schedule.limits.globals)?;
    contract_module.ensure_local_variable_limit(schedule.limits.locals)?;
    contract_module.ensure_parameter_limit(schedule.limits.parameters)?;
    contract_module.ensure_br_table_size_limit(schedule.limits.br_table_size)?;

    if deterministic {
        contract_module.ensure_no_floating_types()?;
    }

    // We disallow importing `gas` function here since it is treated as implementation detail.
    let disallowed_imports = [b"gas".as_ref()];
    let memory_limits =
        get_memory_limits(contract_module.scan_imports(&disallowed_imports)?, schedule)?;

    let code = contract_module
        .inject_gas_metering(deterministic)?
        .inject_stack_height_metering()?
        .into_wasm_code()?;

    Ok((code, memory_limits))
}
